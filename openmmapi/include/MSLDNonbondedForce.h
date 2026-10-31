#ifndef OPENMM_MSLDNONBONDEDFORCE_H_
#define OPENMM_MSLDNONBONDEDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportMSLD.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace MSLDPlugin {

/**
 * Nonbonded (Lennard-Jones + Coulomb) interactions for multisite lambda dynamics.
 *
 * Every particle belongs to a block. Block 0 is the environment and is always present; blocks
 * b >= 1 are added with addBlock() and each belongs to a site. Block b is coupled to the system by
 * the Context global parameter "lambda<b>" (e.g. "lambda1"), which this force declares. The pair
 * coupling L is
 *
 *   both environment                      1   (plain, never soft-cored)
 *   same block b, or environment + b      lambda_b
 *   blocks b, c at different sites        lambda_b * lambda_c
 *   different blocks of the same site     no interaction (excluded, also from PME)
 *
 * Non-environment pairs use the Beutler soft-core form
 *
 *   U_LJ = L 4 eps (1/s^2 - 1/s),            s    = (r/sigma)^6 + alpha (1 - L)
 *   U_C  = L KE q_i q_j g(r_sc),              r_sc = (r^6 + alpha (1 - L) sigma_q^6)^(1/6)
 *
 * with g = 1/r (NoCutoff), 1/r + k_rf r^2 - c_rf (cutoff methods) or erfc(alpha_E r)/r (PME). With
 * PME the reciprocal-space charge of particle i is lambda_b(i) q_i. Exceptions (1-4 pairs) are
 * L (KE qq/r + 4 eps ((sigma/r)^12 - (sigma/r)^6)) with no soft-core and no cutoff; an exception with
 * chargeProd = epsilon = 0 is an exclusion. No long-range dispersion correction is applied.
 *
 * The force reports dU/dlambda_b as an energy parameter derivative for every block b >= 1, so
 * State::getEnergyParameterDerivatives() returns it when parameter derivatives are requested.
 */
class OPENMM_EXPORT_MSLD MSLDNonbondedForce : public OpenMM::Force {
public:
    /**
     * Methods for handling long-range interactions. Values match OpenMM::NonbondedForce.
     */
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        PME = 4
    };
    MSLDNonbondedForce();

    /** Name of the global parameter that couples block b (b >= 1): "lambda<b>". */
    static std::string getLambdaName(int block);

    // ---- blocks ----
    /**
     * Total number of blocks, including the environment (block 0).
     */
    int getNumBlocks() const {
        return blockSite.size();
    }
    /**
     * Add a block at a site (site >= 1). Returns the block index (>= 1).
     *
     * @param site           the site the block belongs to
     * @param defaultLambda  default value of the global parameter lambda<b>
     */
    int addBlock(int site, double defaultLambda = 1.0);
    int getBlockSite(int block) const;
    void setBlockSite(int block, int site);
    double getBlockDefaultLambda(int block) const;
    void setBlockDefaultLambda(int block, double defaultLambda);

    // ---- particles ----
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add a particle. Returns its index.
     *
     * @param charge   charge (e)
     * @param sigma    LJ sigma (nm)
     * @param epsilon  LJ epsilon (kJ/mol)
     * @param block    block the particle belongs to (0 = environment)
     */
    int addParticle(double charge, double sigma, double epsilon, int block = 0);
    void getParticleParameters(int index, double& charge, double& sigma, double& epsilon, int& block) const;
    void setParticleParameters(int index, double charge, double sigma, double epsilon, int block);

    // ---- LJ types and NBFIX ----
    /**
     * Assign a Lennard-Jones type to a particle (-1, the default, means no type). Pairs whose types
     * have an NBFIX entry use its sigma and epsilon instead of Lorentz-Berthelot combining.
     */
    void setParticleLJType(int index, int type);
    int getParticleLJType(int index) const;
    int getNumNBFixPairs() const {
        return nbfix.size();
    }
    /**
     * Add a per-type-pair (sigma, epsilon) override (CHARMM NBFIX). Returns its index.
     */
    int addNBFixPair(int type1, int type2, double sigma, double epsilon);
    void getNBFixPairParameters(int index, int& type1, int& type2, double& sigma, double& epsilon) const;
    void setNBFixPairParameters(int index, int type1, int type2, double sigma, double epsilon);

    // ---- exceptions ----
    int getNumExceptions() const {
        return exceptions.size();
    }
    /**
     * Add an exception (replaces the normal interaction of a pair). chargeProd = epsilon = 0 makes
     * the pair excluded; otherwise it is a 1-4 interaction (no soft-core, no cutoff).
     *
     * @param replace  if true, replace an existing exception for the same pair
     */
    int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace = false);
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const;
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon);

    // ---- method and cutoffs ----
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    bool getUseSwitchingFunction() const;
    void setUseSwitchingFunction(bool use);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    double getReactionFieldDielectric() const;
    void setReactionFieldDielectric(double dielectric);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    /**
     * Explicit PME parameters. alpha = 0 (the default) means "choose from the error tolerance",
     * exactly as OpenMM::NonbondedForce does.
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * The PME parameters actually used by a Context.
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);

    // ---- soft-core ----
    double getSoftcoreAlpha() const;
    /** Soft-core alpha (dimensionless, default 0.5). 0 gives plain linear scaling by L. */
    void setSoftcoreAlpha(double alpha);
    double getSoftcoreSigmaQ() const;
    /** Coulomb soft-core length sigma_q (nm, default 0.3). */
    void setSoftcoreSigmaQ(double sigmaQ);

    /**
     * Copy per-particle, NBFIX and exception parameters to a Context. The set of exception pairs,
     * blocks and particle-to-block assignments cannot change.
     */
    void updateParametersInContext(OpenMM::Context& context);

    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CutoffPeriodic || nonbondedMethod == PME;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    struct ParticleInfo {
        double charge, sigma, epsilon;
        int block, ljType;
    };
    struct ExceptionInfo {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
    };
    struct NBFixInfo {
        int type1, type2;
        double sigma, epsilon;
    };
    NonbondedMethod nonbondedMethod;
    double cutoffDistance, switchingDistance, rfDielectric, ewaldErrorTol, alpha;
    double softcoreAlpha, softcoreSigmaQ;
    bool useSwitchingFunction, exceptionsUsePeriodic;
    int nx, ny, nz;
    std::vector<int> blockSite;
    std::vector<double> blockDefaultLambda;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::vector<NBFixInfo> nbfix;
    std::map<std::pair<int, int>, int> exceptionMap;
};

} // namespace MSLDPlugin

#endif /*OPENMM_MSLDNONBONDEDFORCE_H_*/
