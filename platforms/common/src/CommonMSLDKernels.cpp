/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * GPU (CUDA / OpenCL) implementation of MSLDNonbondedForce on OpenMM's       *
 * common compute platform.  It computes the same energy, forces and          *
 * dU/dlambda_b as the Reference kernel (ReferenceMSLDKernels.cpp):           *
 *   - direct-space pairs through OpenMM's nonbonded tile kernel              *
 *     (soft-core LJ and Coulomb, pair coupling L = lambda_b or               *
 *     lambda_b*lambda_c),                                                    *
 *   - 1-4 exceptions and the PME exclusion corrections through               *
 *     BondedUtilities,                                                       *
 *   - PME reciprocal space on charges lambda_b(i)*q_i, with                  *
 *     dE/dlambda_b = sum_i q_i dE/dQ_i taken from the same grid potential,   *
 *   - self energy and neutralising background on the host (they only        *
 *     depend on per-block charge sums).                                      *
 * -------------------------------------------------------------------------- */

#include "CommonMSLDKernels.h"
#include "CommonMSLDKernelSources.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace MSLDPlugin;
using namespace OpenMM;
using namespace std;

static const int PmeOrder = 5;
static const double KE = ONE_4PI_EPS0;
static const double TWO_OVER_SQRT_PI = 1.1283791670955126;

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

// Tells OpenMM which atoms and pairs are interchangeable when it reorders molecules.
class CommonCalcMSLDNonbondedForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const MSLDNonbondedForce& force, const vector<pair<int, int> >& pairs) : force(force), pairs(pairs) {
        for (int k = 0; k < force.getNumExceptions(); k++) {
            int i, j;
            double q, s, e;
            force.getExceptionParameters(k, i, j, q, s, e);
            exceptionIndex[make_pair(min(i, j), max(i, j))] = k;
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double q1, s1, e1, q2, s2, e2;
        int b1, b2;
        force.getParticleParameters(particle1, q1, s1, e1, b1);
        force.getParticleParameters(particle2, q2, s2, e2, b2);
        return (q1 == q2 && s1 == s2 && e1 == e2 && b1 == b2 && force.getParticleLJType(particle1) == force.getParticleLJType(particle2));
    }
    int getNumParticleGroups() {
        return pairs.size();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.resize(2);
        particles[0] = pairs[index].first;
        particles[1] = pairs[index].second;
    }
    bool areGroupsIdentical(int group1, int group2) {
        map<pair<int, int>, int>::const_iterator e1 = exceptionIndex.find(pairs[group1]);
        map<pair<int, int>, int>::const_iterator e2 = exceptionIndex.find(pairs[group2]);
        if ((e1 == exceptionIndex.end()) != (e2 == exceptionIndex.end()))
            return false;
        if (e1 == exceptionIndex.end())
            return true;
        int i, j;
        double q1, s1, eps1, q2, s2, eps2;
        force.getExceptionParameters(e1->second, i, j, q1, s1, eps1);
        force.getExceptionParameters(e2->second, i, j, q2, s2, eps2);
        return (q1 == q2 && s1 == s2 && eps1 == eps2);
    }
private:
    const MSLDNonbondedForce& force;
    const vector<pair<int, int> >& pairs;
    map<pair<int, int>, int> exceptionIndex;
};

CommonCalcMSLDNonbondedForceKernel::~CommonCalcMSLDNonbondedForceKernel() {
}

string CommonCalcMSLDNonbondedForceKernel::addDerivative(const vector<string>& derivVariables, const string& blockVar, const string& value) const {
    stringstream s;
    for (int b = 1; b < numBlocks; b++) {
        if (b > 1)
            s << "else ";
        s << "if (" << blockVar << " == " << b << ") " << derivVariables[b] << " += " << value << ";\n";
    }
    return s.str();
}

void CommonCalcMSLDNonbondedForceKernel::initialize(const System& system, const MSLDNonbondedForce& force) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("MSLDNonbondedForce: running on more than one device is not supported");
    numParticles = force.getNumParticles();
    numBlocks = force.getNumBlocks();
    if (numBlocks > 1023)
        throw OpenMMException("MSLDNonbondedForce: at most 1023 blocks are supported on this platform");
    blockSite.resize(numBlocks);
    lambdaNames.resize(numBlocks);
    lambdaGlobalIndex.assign(numBlocks, 0);
    for (int b = 0; b < numBlocks; b++) {
        blockSite[b] = force.getBlockSite(b);
        if (b > 0) {
            lambdaNames[b] = MSLDNonbondedForce::getLambdaName(b);
            lambdaGlobalIndex[b] = cc.registerGlobalParam(lambdaNames[b]);
            cc.addEnergyParameterDerivative(lambdaNames[b]);
        }
    }
    block.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double q, s, e;
        force.getParticleParameters(i, q, s, e, block[i]);
    }
    method = force.getNonbondedMethod();
    bool useCutoff = (method != MSLDNonbondedForce::NoCutoff);
    bool usePeriodic = (method == MSLDNonbondedForce::CutoffPeriodic || method == MSLDNonbondedForce::PME);
    usePME = (method == MSLDNonbondedForce::PME);
    double cutoff = force.getCutoffDistance();

    // Exclusions: every exception pair, plus every pair between different blocks of the same site.
    set<pair<int, int> > exclusionSet;
    exception14Index.clear();
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int i, j;
        double q, s, e;
        force.getExceptionParameters(k, i, j, q, s, e);
        exclusionSet.insert(make_pair(min(i, j), max(i, j)));
        if (q != 0.0 || e != 0.0)
            exception14Index.push_back(k);
    }
    numExceptions14 = exception14Index.size();
    exclusionPairs.assign(exclusionSet.begin(), exclusionSet.end());
    pmeExclusions = exclusionPairs;
    vector<int> alchemical;
    for (int i = 0; i < numParticles; i++)
        if (block[i] != 0)
            alchemical.push_back(i);
    for (int m = 0; m < (int) alchemical.size(); m++)
        for (int n = m+1; n < (int) alchemical.size(); n++) {
            int i = alchemical[m], j = alchemical[n];
            if (block[i] != block[j] && blockSite[block[i]] == blockSite[block[j]] && exclusionSet.find(make_pair(i, j)) == exclusionSet.end())
                pmeExclusions.push_back(make_pair(i, j));
        }
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++)
        exclusionList[i].push_back(i);
    for (auto& p : pmeExclusions) {
        exclusionList[p.first].push_back(p.second);
        exclusionList[p.second].push_back(p.first);
    }

    // LJ types used by NBFix pairs.
    numTypes = 0;
    if (force.getNumNBFixPairs() > 0) {
        for (int i = 0; i < numParticles; i++)
            numTypes = max(numTypes, force.getParticleLJType(i)+1);
        for (int k = 0; k < force.getNumNBFixPairs(); k++) {
            int t1, t2;
            double s, e;
            force.getNBFixPairParameters(k, t1, t2, s, e);
            numTypes = max(numTypes, max(t1, t2)+1);
        }
    }

    // Device arrays and parameters.
    particleParams.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "msldParticleParams");
    nbfixTable.initialize<mm_float2>(cc, max(1, numTypes*numTypes), "msldNBFix");
    lambdaIndex.initialize<int>(cc, numBlocks, "msldLambdaIndex");
    lambdaIndex.upload(lambdaGlobalIndex);
    exceptionParams.initialize<mm_float4>(cc, max(1, numExceptions14), "msldExceptionParams");
    uploadParameters(force);

    // PME parameters, chosen the way NonbondedForce chooses them.
    ewaldAlpha = 0.0;
    gridSize[0] = gridSize[1] = gridSize[2] = 0;
    if (usePME) {
        NonbondedForce equivalent;
        equivalent.setNonbondedMethod(NonbondedForce::PME);
        equivalent.setCutoffDistance(cutoff);
        equivalent.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
        double alpha;
        int nx, ny, nz;
        force.getPMEParameters(alpha, nx, ny, nz);
        equivalent.setPMEParameters(alpha, nx, ny, nz);
        for (int i = 0; i < numParticles; i++) {
            double q, s, e;
            int b;
            force.getParticleParameters(i, q, s, e, b);
            equivalent.addParticle(q, s, e);
        }
        NonbondedForceImpl::calcPMEParameters(system, equivalent, ewaldAlpha, gridSize[0], gridSize[1], gridSize[2], false);
        for (int d = 0; d < 3; d++)
            gridSize[d] = cc.findLegalFFTDimension(gridSize[d]);
    }

    // ---- direct-space pairs ----
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    vector<string> nbDerivs(numBlocks);
    for (int b = 1; b < numBlocks; b++)
        nbDerivs[b] = nb.addEnergyParameterDerivative(lambdaNames[b]);
    map<string, string> structure, constants;
    structure["CUTOFF_TEST"] = (useCutoff ? "!isExcluded && r2 < (real) "+cc.doubleToString(cutoff*cutoff) : "!isExcluded");
    if (numTypes > 0)
        structure["LJ_PAIR"] =
            "int t1 = (bw1 >> 10) - 1, t2 = (bw2 >> 10) - 1;\n"
            "float2 fix = (t1 >= 0 && t2 >= 0 ? msldNBFix[t1*"+cc.intToString(numTypes)+"+t2] : make_float2(-1.0f, 0.0f));\n"
            "if (fix.x >= 0) { sig = fix.x; eps = fix.y; }\n"
            "else { sig = 0.5f*(mp1.y+mp2.y); eps = mp1.z*mp2.z; }\n";
    else
        structure["LJ_PAIR"] = "sig = 0.5f*(mp1.y+mp2.y);\neps = mp1.z*mp2.z;\n";
    if (useCutoff && force.getUseSwitchingFunction()) {
        double sw = force.getSwitchingDistance();
        string swd = "(real) "+cc.doubleToString(sw);
        string inv = "(real) "+cc.doubleToString(1.0/(cutoff-sw));
        structure["SWITCH_CODE"] =
            "if (r > "+swd+") {\n"
            "    real t = (r-"+swd+")*"+inv+";\n"
            "    sw = 1+t*t*t*(-10+t*(15-t*6));\n"
            "    dsw = t*t*(-30+t*(60-t*30))*"+inv+";\n"
            "}\n";
    }
    else
        structure["SWITCH_CODE"] = "";
    if (method == MSLDNonbondedForce::PME)
        structure["COULOMB_KERNEL"] =
            "real ar = (real) EWALD_ALPHA*rsc;\n"
            "real expAr2 = EXP(-ar*ar);\n"
            "#ifdef USE_DOUBLE_PRECISION\n"
            "real erfcAr = erfc(ar);\n"
            "#else\n"
            "real tt = RECIP(1.0f+0.3275911f*ar);\n"
            "real erfcAr = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*tt)*tt)*tt)*tt)*tt*expAr2;\n"
            "#endif\n"
            "g = erfcAr*invRsc;\n"
            "dg = -(erfcAr*invRsc + (real) (TWO_OVER_SQRT_PI*EWALD_ALPHA)*expAr2)*invRsc;\n";
    else if (method == MSLDNonbondedForce::NoCutoff)
        structure["COULOMB_KERNEL"] = "g = invRsc;\ndg = -invRsc*invRsc;\n";
    else {
        double eps = force.getReactionFieldDielectric();
        double krf = pow(cutoff, -3.0)*(eps-1.0)/(2.0*eps+1.0);
        double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
        structure["COULOMB_KERNEL"] =
            "g = invRsc + (real) "+cc.doubleToString(krf)+"*rsc*rsc - (real) "+cc.doubleToString(crf)+";\n"
            "dg = -invRsc*invRsc + (real) "+cc.doubleToString(2*krf)+"*rsc;\n";
    }
    structure["ADD_DERIV_A"] = addDerivative(nbDerivs, "ba", "interactionScale*dUdL*dL1");
    structure["ADD_DERIV_B"] = addDerivative(nbDerivs, "bb", "interactionScale*dUdL*dL2");
    constants["SC_ALPHA"] = cc.doubleToString(force.getSoftcoreAlpha());
    constants["SIGQ6"] = cc.doubleToString(pow(force.getSoftcoreSigmaQ(), 6.0));
    constants["KE"] = cc.doubleToString(KE);
    constants["EWALD_ALPHA"] = cc.doubleToString(ewaldAlpha);
    constants["TWO_OVER_SQRT_PI"] = cc.doubleToString(TWO_OVER_SQRT_PI);
    string pairSource = cc.replaceStrings(cc.replaceStrings(CommonMSLDKernelSources::pairInteraction, structure), constants);
    nb.addInteraction(useCutoff, usePeriodic, true, cutoff, exclusionList, pairSource, force.getForceGroup());
    nb.addParameter(ComputeParameterInfo(particleParams, "msldParams", "float", 4));
    nb.addArgument(ComputeParameterInfo(cc.getGlobalParamValues(), "msldGlobals", "real", 1));
    nb.addArgument(ComputeParameterInfo(lambdaIndex, "msldLambdaIndex", "int", 1));
    if (numTypes > 0)
        nb.addArgument(ComputeParameterInfo(nbfixTable, "msldNBFix", "float", 2));

    // ---- 1-4 exceptions ----
    BondedUtilities& bonded = cc.getBondedUtilities();
    if (numExceptions14 > 0) {
        vector<vector<int> > atoms(numExceptions14, vector<int>(2));
        for (int k = 0; k < numExceptions14; k++) {
            double q, s, e;
            force.getExceptionParameters(exception14Index[k], atoms[k][0], atoms[k][1], q, s, e);
        }
        vector<string> derivs(numBlocks);
        for (int b = 1; b < numBlocks; b++)
            derivs[b] = bonded.addEnergyParameterDerivative(lambdaNames[b]);
        map<string, string> rep;
        rep["EXC_PARAMS"] = bonded.addArgument(exceptionParams, "float4");
        rep["GLOBALS"] = bonded.addArgument(cc.getGlobalParamValues(), "real");
        rep["LAMBDA_INDEX"] = bonded.addArgument(lambdaIndex, "int");
        rep["USE_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        rep["ADD_DERIV"] = addDerivative(derivs, "bx", "u");
        bonded.addInteraction(atoms, cc.replaceStrings(CommonMSLDKernelSources::exception14, rep), force.getForceGroup());
    }

    // ---- PME ----
    if (usePME) {
        if (pmeExclusions.size() > 0) {
            vector<vector<int> > atoms(pmeExclusions.size(), vector<int>(2));
            for (int k = 0; k < (int) pmeExclusions.size(); k++) {
                atoms[k][0] = pmeExclusions[k].first;
                atoms[k][1] = pmeExclusions[k].second;
            }
            vector<string> derivs(numBlocks);
            for (int b = 1; b < numBlocks; b++)
                derivs[b] = bonded.addEnergyParameterDerivative(lambdaNames[b]);
            map<string, string> rep;
            rep["PARAMS"] = bonded.addArgument(particleParams, "float4");
            rep["GLOBALS"] = bonded.addArgument(cc.getGlobalParamValues(), "real");
            rep["LAMBDA_INDEX"] = bonded.addArgument(lambdaIndex, "int");
            rep["ADD_DERIV_A"] = addDerivative(derivs, "ba", "-(real) KE*pa.x*Qb*erfOverR");
            rep["ADD_DERIV_B"] = addDerivative(derivs, "bb", "-(real) KE*Qa*pb.x*erfOverR");
            string source = cc.replaceStrings(cc.replaceStrings(CommonMSLDKernelSources::pmeExclusion, rep), constants);
            bonded.addInteraction(atoms, source, force.getForceGroup());
        }
        int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        int gridElements = gridSize[0]*gridSize[1]*gridSize[2];
        useFixedPointChargeSpreading = cc.getSupports64BitGlobalAtomics();
        charges.initialize(cc, cc.getPaddedNumAtoms(), elementSize, "msldCharges");
        pmeGrid1.initialize(cc, gridElements, 2*elementSize, "msldPmeGrid1");
        pmeGrid2.initialize(cc, gridElements, 2*elementSize, "msldPmeGrid2");
        if (useFixedPointChargeSpreading)
            cc.addAutoclearBuffer(pmeGrid2);
        else
            cc.addAutoclearBuffer(pmeGrid1);
        derivIndex.initialize<int>(cc, numBlocks, "msldDerivIndex");

        // b-spline moduli, as in OpenMM's CommonCalcNonbondedForceKernel
        int maxSize = max(max(gridSize[0], gridSize[1]), gridSize[2]);
        vector<double> data(PmeOrder), bsplinesData(maxSize+1, 0.0);
        data[PmeOrder-1] = 0.0;
        data[1] = 0.0;
        data[0] = 1.0;
        for (int i = 3; i < PmeOrder; i++) {
            double div = 1.0/(i-1.0);
            data[i-1] = 0.0;
            for (int j = 1; j < (i-1); j++)
                data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
            data[0] = div*data[0];
        }
        double div = 1.0/(PmeOrder-1);
        data[PmeOrder-1] = 0.0;
        for (int i = 1; i < (PmeOrder-1); i++)
            data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
        data[0] = div*data[0];
        for (int i = 1; i <= PmeOrder; i++)
            bsplinesData[i] = data[i-1];
        ComputeArray* moduliArrays[3] = {&moduliX, &moduliY, &moduliZ};
        for (int dim = 0; dim < 3; dim++) {
            int ndata = gridSize[dim];
            vector<double> moduli(ndata);
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0, ss = 0.0;
                for (int j = 0; j < ndata; j++) {
                    double arg = (2.0*M_PI*i*j)/ndata;
                    sc += bsplinesData[j]*cos(arg);
                    ss += bsplinesData[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }
            for (int i = 0; i < ndata; i++)
                if (moduli[i] < 1.0e-7)
                    moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
            moduliArrays[dim]->initialize(cc, ndata, elementSize, "msldPmeModuli");
            moduliArrays[dim]->upload(moduli, true);
        }
        fft = cc.createFFT(gridSize[0], gridSize[1], gridSize[2], true);
    }
    info = new ForceInfo(force, pmeExclusions);
    cc.addForce(info);
}

void CommonCalcMSLDNonbondedForceKernel::uploadParameters(const MSLDNonbondedForce& force) {
    vector<mm_float4> params(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    blockChargeSum.assign(numBlocks, 0.0);
    blockChargeSquaredSum.assign(numBlocks, 0.0);
    for (int i = 0; i < numParticles; i++) {
        double q, s, e;
        int b;
        force.getParticleParameters(i, q, s, e, b);
        if (b != block[i])
            throw OpenMMException("updateParametersInContext: the block of a particle cannot be changed");
        int type = (numTypes > 0 ? force.getParticleLJType(i) : -1);
        if (type >= numTypes)
            throw OpenMMException("updateParametersInContext: the number of LJ types cannot grow");
        params[i] = mm_float4((float) q, (float) s, (float) sqrt(e), (float) (b + 1024*(type+1)));
        blockChargeSum[b] += q;
        blockChargeSquaredSum[b] += q*q;
    }
    particleParams.upload(params);
    if (numTypes > 0) {
        vector<mm_float2> table(numTypes*numTypes, mm_float2(-1.0f, 0.0f));
        for (int k = 0; k < force.getNumNBFixPairs(); k++) {
            int t1, t2;
            double s, e;
            force.getNBFixPairParameters(k, t1, t2, s, e);
            if (t1 >= numTypes || t2 >= numTypes)
                throw OpenMMException("updateParametersInContext: the number of LJ types cannot grow");
            table[t1*numTypes+t2] = table[t2*numTypes+t1] = mm_float2((float) s, (float) e);
        }
        nbfixTable.upload(table);
    }
    set<pair<int, int> > newExclusions;
    vector<int> new14;
    for (int k = 0; k < force.getNumExceptions(); k++) {
        int i, j;
        double q, s, e;
        force.getExceptionParameters(k, i, j, q, s, e);
        newExclusions.insert(make_pair(min(i, j), max(i, j)));
        if (q != 0.0 || e != 0.0)
            new14.push_back(k);
    }
    if (vector<pair<int, int> >(newExclusions.begin(), newExclusions.end()) != exclusionPairs || new14 != exception14Index)
        throw OpenMMException("updateParametersInContext: the set of exceptions has changed");
    if (numExceptions14 > 0) {
        vector<mm_float4> excParams(numExceptions14);
        for (int k = 0; k < numExceptions14; k++) {
            int i, j;
            double q, s, e;
            force.getExceptionParameters(exception14Index[k], i, j, q, s, e);
            int b = (block[i] != 0 ? block[i] : block[j]);
            excParams[k] = mm_float4((float) (KE*q), (float) s, (float) (4.0*e), (float) b);
        }
        exceptionParams.upload(excParams);
    }
}

void CommonCalcMSLDNonbondedForceKernel::initializeKernels() {
    hasInitializedKernels = true;
    if (!usePME)
        return;
    const vector<string>& derivNames = cc.getEnergyParamDerivNames();
    vector<int> index(numBlocks, 0);
    for (int b = 1; b < numBlocks; b++)
        for (int k = 0; k < (int) derivNames.size(); k++)
            if (derivNames[k] == lambdaNames[b])
                index[b] = k;
    derivIndex.upload(index);
    map<string, string> defines;
    defines["PME_ORDER"] = cc.intToString(PmeOrder);
    defines["NUM_ATOMS"] = cc.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["RECIP_EXP_FACTOR"] = cc.doubleToString(M_PI*M_PI/(ewaldAlpha*ewaldAlpha));
    defines["GRID_SIZE_X"] = cc.intToString(gridSize[0]);
    defines["GRID_SIZE_Y"] = cc.intToString(gridSize[1]);
    defines["GRID_SIZE_Z"] = cc.intToString(gridSize[2]);
    defines["EPSILON_FACTOR"] = cc.doubleToString(sqrt(KE));
    defines["M_PI"] = cc.doubleToString(M_PI);
    defines["NUM_DERIVS"] = cc.intToString(numBlocks > 1 ? derivNames.size() : 0);
    if (useFixedPointChargeSpreading)
        defines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
    ComputeProgram program = cc.compileProgram(CommonMSLDKernelSources::pme, defines);
    chargesKernel = program->createKernel("computeCharges");
    chargesKernel->addArg(particleParams);
    chargesKernel->addArg(cc.getGlobalParamValues());
    chargesKernel->addArg(lambdaIndex);
    chargesKernel->addArg(charges);
    spreadKernel = program->createKernel("gridSpreadCharge");
    spreadKernel->addArg(cc.getPosq());
    spreadKernel->addArg(useFixedPointChargeSpreading ? pmeGrid2 : pmeGrid1);
    for (int i = 0; i < 8; i++)
        spreadKernel->addArg();
    spreadKernel->addArg(charges);
    if (useFixedPointChargeSpreading) {
        finishSpreadKernel = program->createKernel("finishSpreadCharge");
        finishSpreadKernel->addArg(pmeGrid2);
        finishSpreadKernel->addArg(pmeGrid1);
    }
    convolutionKernel = program->createKernel("reciprocalConvolution");
    convolutionKernel->addArg(pmeGrid2);
    convolutionKernel->addArg(moduliX);
    convolutionKernel->addArg(moduliY);
    convolutionKernel->addArg(moduliZ);
    for (int i = 0; i < 3; i++)
        convolutionKernel->addArg();
    energyKernel = program->createKernel("gridEvaluateEnergy");
    energyKernel->addArg(pmeGrid2);
    energyKernel->addArg(cc.getEnergyBuffer());
    energyKernel->addArg(moduliX);
    energyKernel->addArg(moduliY);
    energyKernel->addArg(moduliZ);
    for (int i = 0; i < 3; i++)
        energyKernel->addArg();
    interpolateKernel = program->createKernel("gridInterpolateForce");
    interpolateKernel->addArg(cc.getPosq());
    interpolateKernel->addArg(cc.getLongForceBuffer());
    interpolateKernel->addArg(pmeGrid1);
    for (int i = 0; i < 8; i++)
        interpolateKernel->addArg();
    interpolateKernel->addArg(charges);
    interpolateKernel->addArg(particleParams);
    interpolateKernel->addArg(derivIndex);
    if (numBlocks > 1)
        interpolateKernel->addArg(cc.getEnergyParamDerivBuffer());
}

double CommonCalcMSLDNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    if (!hasInitializedKernels)
        initializeKernels();
    if (!usePME)
        return 0.0;
    Vec3 box[3];
    cc.getPeriodicBoxVectors(box[0], box[1], box[2]);
    double determinant = box[0][0]*box[1][1]*box[2][2];
    double scale = 1.0/determinant;
    mm_double4 recip[3];
    recip[0] = mm_double4(box[1][1]*box[2][2]*scale, 0, 0, 0);
    recip[1] = mm_double4(-box[1][0]*box[2][2]*scale, box[0][0]*box[2][2]*scale, 0, 0);
    recip[2] = mm_double4((box[1][0]*box[2][1]-box[1][1]*box[2][0])*scale, -box[0][0]*box[2][1]*scale, box[0][0]*box[1][1]*scale, 0);
    mm_float4 recipFloat[3];
    for (int i = 0; i < 3; i++)
        recipFloat[i] = mm_float4((float) recip[i].x, (float) recip[i].y, (float) recip[i].z, 0);
    bool useDouble = cc.getUseDoublePrecision();
    auto setRecip = [&](ComputeKernel& kernel, int index) {
        for (int i = 0; i < 3; i++) {
            if (useDouble)
                kernel->setArg(index+i, recip[i]);
            else
                kernel->setArg(index+i, recipFloat[i]);
        }
    };
    int gridPoints = gridSize[0]*gridSize[1]*gridSize[2];
    chargesKernel->execute(numParticles);
    setPeriodicBoxArgs(cc, spreadKernel, 2);
    setRecip(spreadKernel, 7);
    spreadKernel->execute(numParticles*PmeOrder);
    if (useFixedPointChargeSpreading)
        finishSpreadKernel->execute(gridPoints);
    fft->execFFT(pmeGrid1, pmeGrid2, true);
    if (includeEnergy) {
        setRecip(energyKernel, 5);
        energyKernel->execute(gridPoints);
    }
    setRecip(convolutionKernel, 4);
    convolutionKernel->execute(gridPoints);
    fft->execFFT(pmeGrid2, pmeGrid1, false);
    setPeriodicBoxArgs(cc, interpolateKernel, 3);
    setRecip(interpolateKernel, 8);
    interpolateKernel->execute(numParticles);

    // Self energy and neutralising background depend only on per-block charge sums.
    vector<double> lam(numBlocks, 1.0);
    for (int b = 1; b < numBlocks; b++)
        lam[b] = context.getParameter(lambdaNames[b]);
    double selfScale = KE*ewaldAlpha/sqrt(M_PI);
    double volume = box[0][0]*box[1][1]*box[2][2];
    double bgScale = KE*M_PI/(volume*ewaldAlpha*ewaldAlpha);
    double totalCharge = 0.0, energy = 0.0;
    for (int b = 0; b < numBlocks; b++) {
        totalCharge += lam[b]*blockChargeSum[b];
        energy -= selfScale*lam[b]*lam[b]*blockChargeSquaredSum[b];
    }
    energy -= 0.5*bgScale*totalCharge*totalCharge;
    map<string, double>& derivs = cc.getEnergyParamDerivWorkspace();
    for (int b = 1; b < numBlocks; b++)
        derivs[lambdaNames[b]] += -2.0*selfScale*lam[b]*blockChargeSquaredSum[b] - bgScale*totalCharge*blockChargeSum[b];
    return energy;
}

void CommonCalcMSLDNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const MSLDNonbondedForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: the number of particles has changed");
    if (force.getNumBlocks() != numBlocks)
        throw OpenMMException("updateParametersInContext: the number of blocks has changed");
    for (int b = 1; b < numBlocks; b++)
        if (force.getBlockSite(b) != blockSite[b])
            throw OpenMMException("updateParametersInContext: the site of a block cannot be changed");
    uploadParameters(force);
    cc.invalidateMolecules(info);
}

void CommonCalcMSLDNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (method != MSLDNonbondedForce::PME)
        throw OpenMMException("getPMEParametersInContext: this Context is not using PME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}
