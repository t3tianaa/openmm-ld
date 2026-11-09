%module msldplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_string.i"
%include "typemaps.i"

%{
#include "MSLDNonbondedForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
%}

/*
 * Plain floats in and out, in OpenMM's internal units (nm, kJ/mol, e).
 */

namespace MSLDPlugin {

%apply double& OUTPUT { double& charge, double& sigma, double& epsilon, double& chargeProd, double& alpha };
%apply int& OUTPUT { int& block, int& particle1, int& particle2, int& type1, int& type2, int& nx, int& ny, int& nz };

class MSLDNonbondedForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        PME = 4
    };
    MSLDNonbondedForce();
    static std::string getLambdaName(int block);

    int getNumBlocks() const;
    int addBlock(int site, double defaultLambda = 1.0);
    int getBlockSite(int block) const;
    void setBlockSite(int block, int site);
    double getBlockDefaultLambda(int block) const;
    void setBlockDefaultLambda(int block, double defaultLambda);

    int getNumParticles() const;
    int addParticle(double charge, double sigma, double epsilon, int block = 0);
    void getParticleParameters(int index, double& charge, double& sigma, double& epsilon, int& block) const;
    void setParticleParameters(int index, double charge, double sigma, double epsilon, int block);

    void setParticleLJType(int index, int type);
    int getParticleLJType(int index) const;
    int getNumNBFixPairs() const;
    int addNBFixPair(int type1, int type2, double sigma, double epsilon);
    void getNBFixPairParameters(int index, int& type1, int& type2, double& sigma, double& epsilon) const;
    void setNBFixPairParameters(int index, int type1, int type2, double sigma, double epsilon);

    int getNumExceptions() const;
    int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace = false);
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const;
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon);

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
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);

    double getSoftcoreAlpha() const;
    void setSoftcoreAlpha(double alpha);
    double getSoftcoreSigmaQ() const;
    void setSoftcoreSigmaQ(double sigmaQ);

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    %extend {
        static MSLDPlugin::MSLDNonbondedForce& cast(OpenMM::Force& force) {
            return dynamic_cast<MSLDPlugin::MSLDNonbondedForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<MSLDPlugin::MSLDNonbondedForce*>(&force) != NULL);
        }
    }
};

%clear double& charge, double& sigma, double& epsilon, double& chargeProd, double& alpha;
%clear int& block, int& particle1, int& particle2, int& type1, int& type2, int& nx, int& ny, int& nz;

}

%pythoncode %{
def loadReferencePlugin(path):
    """Load the Reference/CPU kernel library (libOpenMMMSLDReference) from an explicit path, for
    use without installing it into OpenMM's plugins directory."""
    mm.Platform.loadPluginLibrary(path)
%}
