#ifndef COMMON_MSLD_KERNEL_SOURCES_H_
#define COMMON_MSLD_KERNEL_SOURCES_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM MSLD plugin                                 *
 *                                                                            *
 * Device code for the common (CUDA / OpenCL) MSLDNonbondedForce kernel.      *
 *                                                                            *
 * The PME kernels are adapted from OpenMM's                                  *
 * platforms/common/src/kernels/pme.cc, Copyright (c) Stanford University and *
 * the Authors, distributed under the GNU Lesser General Public License v3.   *
 * Changes: charges are lambda_b(i)*q_i from a per-atom array, atoms are not  *
 * sorted, and the interpolation kernel also accumulates dE/dlambda_b.        *
 * -------------------------------------------------------------------------- */

namespace MSLDPlugin {
namespace CommonMSLDKernelSources {

// Direct-space pair interaction, inserted into OpenMM's nonbonded kernel.
// Placeholders (replaced on the host): CUTOFF_TEST, LJ_PAIR, COULOMB_KERNEL, ADD_DERIV_A, ADD_DERIV_B,
// SC_ALPHA, SIGQ6, KE, SWITCH_CODE.
static const char* pairInteraction = R"MSLD(
{
    float4 mp1 = msldParams1;
    float4 mp2 = msldParams2;
    int bw1 = (int) mp1.w;
    int bw2 = (int) mp2.w;
    int b1 = bw1 & 1023;
    int b2 = bw2 & 1023;
    if (CUTOFF_TEST) {
        real lam1 = (b1 == 0 ? (real) 1 : msldGlobals[msldLambdaIndex[b1]]);
        real lam2 = (b2 == 0 ? (real) 1 : msldGlobals[msldLambdaIndex[b2]]);
        real L = 1, dL1 = 0, dL2 = 0;
        int ba = 0, bb = 0;
        if (b1 == 0 && b2 == 0) {
        }
        else if (b1 == b2 || b1 == 0 || b2 == 0) {
            ba = (b1 != 0 ? b1 : b2);
            L = (b1 != 0 ? lam1 : lam2);
            dL1 = 1;
        }
        else {
            ba = b1;
            bb = b2;
            L = lam1*lam2;
            dL1 = lam2;
            dL2 = lam1;
        }
        real shift = (ba != 0 ? (real) SC_ALPHA*(1-L) : (real) 0);
        real r6 = r2*r2*r2;
        real U = 0, dUdr = 0, dUdL = 0;

        // Lennard-Jones, Beutler soft-core
        real sig, eps;
        LJ_PAIR
        if (eps != 0 && sig != 0) {
            real sw = 1, dsw = 0;
            SWITCH_CODE
            real sig2 = sig*sig;
            real sig6 = sig2*sig2*sig2;
            real s = r6/sig6 + shift;
            real inv = RECIP(s);
            real f = inv*inv - inv;
            real fp = inv*inv*(1 - 2*inv);
            real ulj = L*4*eps*f;
            real duljdr = L*4*eps*fp*6*r6*invR/sig6;
            U += ulj*sw;
            dUdr += duljdr*sw + ulj*dsw;
            dUdL += 4*eps*(f - (real) SC_ALPHA*L*fp)*sw;
        }

        // Coulomb, soft-core on the distance
        real qq = (real) KE*mp1.x*mp2.x;
        if (qq != 0) {
            real rsc = (shift > 0 ? POW(r6 + shift*(real) SIGQ6, (real) (1.0/6.0)) : r);
            real invRsc = RECIP(rsc);
            real g, dg;
            COULOMB_KERNEL
            real rsc2 = invRsc*invRsc;
            real invRsc5 = rsc2*rsc2*invRsc;
            U += L*qq*g;
            dUdr += L*qq*dg*(r6*invR)*invRsc5;
            dUdL += qq*(g - L*dg*(real) (SC_ALPHA*SIGQ6/6.0)*invRsc5);
        }
        tempEnergy += U;
        dEdR -= dUdr*invR;
        ADD_DERIV_A
        ADD_DERIV_B
    }
}
)MSLD";

// 1-4 exceptions: L*(KE qq/r + LJ), no soft-core, no cutoff.  Run by BondedUtilities.
static const char* exception14 = R"MSLD(
float4 ep = EXC_PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if USE_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real u = ep.x*invR;
real dudr = -u*invR;
if (ep.z != 0) {
    real sr2 = ep.y*ep.y*invR*invR;
    real sr6 = sr2*sr2*sr2;
    u += ep.z*(sr6*sr6-sr6);
    dudr += ep.z*(-12*sr6*sr6+6*sr6)*invR;
}
int bx = (int) ep.w;
real L = (bx == 0 ? (real) 1 : GLOBALS[LAMBDA_INDEX[bx]]);
energy += L*u;
delta *= L*dudr*invR;
real3 force1 = delta;
real3 force2 = -delta;
ADD_DERIV
)MSLD";

// Removes the reciprocal-space interaction of excluded pairs (exceptions and same-site pairs of
// different blocks).  Run by BondedUtilities.
static const char* pmeExclusion = R"MSLD(
float4 pa = PARAMS[atom1];
float4 pb = PARAMS[atom2];
int ba = ((int) pa.w) & 1023;
int bb = ((int) pb.w) & 1023;
real Qa = (ba == 0 ? (real) 1 : GLOBALS[LAMBDA_INDEX[ba]])*pa.x;
real Qb = (bb == 0 ? (real) 1 : GLOBALS[LAMBDA_INDEX[bb]])*pb.x;
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
APPLY_PERIODIC_TO_DELTA(delta)
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real r = SQRT(r2);
real alphaR = (real) EWALD_ALPHA*r;
real erfOverR;
real3 force1 = make_real3(0);
real3 force2 = make_real3(0);
if (alphaR > 1e-6f) {
    real invR = RECIP(r);
    erfOverR = ERF(alphaR)*invR;
    real dEdr = -(real) KE*Qa*Qb*((real) (TWO_OVER_SQRT_PI*EWALD_ALPHA)*EXP(-alphaR*alphaR) - erfOverR)*invR;
    force1 = delta*(dEdr*invR);
    force2 = -force1;
}
else
    erfOverR = (real) (TWO_OVER_SQRT_PI*EWALD_ALPHA);
energy -= (real) KE*Qa*Qb*erfOverR;
ADD_DERIV_A
ADD_DERIV_B
)MSLD";

static const char* pme = R"MSLD(
KERNEL void computeCharges(GLOBAL const float4* RESTRICT params, GLOBAL const real* RESTRICT globals,
        GLOBAL const int* RESTRICT lambdaIndex, GLOBAL real* RESTRICT charges) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        float4 p = params[atom];
        int b = ((int) p.w) & 1023;
        charges[atom] = (b == 0 ? (real) 1 : globals[lambdaIndex[b]])*p.x;
    }
}

KERNEL void gridSpreadCharge(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real* RESTRICT pmeGrid,
#endif
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const real* RESTRICT charges) {
    real3 data[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));
    for (int i = GLOBAL_ID; i < NUM_ATOMS*PME_ORDER; i += GLOBAL_SIZE) {
        int atom = i/PME_ORDER;
        const real charge = charges[atom]*EPSILON_FACTOR;
        if (charge == 0)
            continue;
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
        int iz = i%PME_ORDER;
        int zindex = gridIndex.z+iz;
        zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
        real dz = 0;
        for (int j = 0; j < PME_ORDER; j++)
            dz = (j == iz ? data[j].z : dz);
        dz *= charge;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dzdx = dz*data[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = ybase*GRID_SIZE_Z;
                int index = xbase + ybase + zindex;
                real add = dzdx*data[iy].y;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                if (fabs(add) > 2.3e-10f)
                    ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add));
#else
                ATOMIC_ADD(&pmeGrid[index], add);
#endif
            }
        }
    }
}

#ifdef USE_FIXED_POINT_CHARGE_SPREADING
KERNEL void finishSpreadCharge(GLOBAL const mm_long* RESTRICT grid1, GLOBAL real* RESTRICT grid2) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (unsigned int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE)
        grid2[index] = scale*grid1[index];
}
#endif

KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
    const real recipScaleFactor = RECIP((real) MSLD_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    for (unsigned int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
        int ky = remainder/(GRID_SIZE_Z/2+1);
        int kz = remainder-ky*(GRID_SIZE_Z/2+1);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
        if (kx != 0 || ky != 0 || kz != 0)
            pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
        else
            pmeGrid[index] = make_real2(0);
    }
}

KERNEL void gridEvaluateEnergy(GLOBAL real2* RESTRICT pmeGrid, GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real* RESTRICT pmeBsplineModuliX, GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    const real recipScaleFactor = RECIP((real) MSLD_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    mixed energy = 0;
    for (unsigned int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z);
        int ky = remainder/(GRID_SIZE_Z);
        int kz = remainder-ky*(GRID_SIZE_Z);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
        if (kz >= (GRID_SIZE_Z/2+1)) {
            kx = ((kx == 0) ? kx : GRID_SIZE_X-kx);
            ky = ((ky == 0) ? ky : GRID_SIZE_Y-ky);
            kz = GRID_SIZE_Z-kz;
        }
        int indexInHalfComplexGrid = kz + ky*(GRID_SIZE_Z/2+1)+kx*(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        real2 grid = pmeGrid[indexInHalfComplexGrid];
        if (kx != 0 || ky != 0 || kz != 0)
            energy += eterm*(grid.x*grid.x + grid.y*grid.y);
    }
    energyBuffer[GLOBAL_ID] += 0.5f*energy;
}

// Forces from the reciprocal-space potential, and dE/dlambda_b = sum_i q_i dE/dQ_i.
KERNEL void gridInterpolateForce(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const real* RESTRICT charges,
        GLOBAL const float4* RESTRICT params, GLOBAL const int* RESTRICT derivIndex
#if NUM_DERIVS > 0
        , GLOBAL mixed* RESTRICT energyParamDerivs
#endif
        ) {
    real3 data[PME_ORDER];
    real3 ddata[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        float4 p = params[atom];
        int b = ((int) p.w) & 1023;
        real q = charges[atom]*EPSILON_FACTOR;
        if (p.x == 0 || (q == 0 && b == 0))
            continue;
        real3 force = make_real3(0);
        real phi = 0;
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        ddata[0] = -data[0];
        for (int j = 1; j < PME_ORDER; j++)
            ddata[j] = data[j-1]-data[j];
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    real gridvalue = pmeGrid[ybase + zindex];
                    force.x += ddx*dy*data[iz].z*gridvalue;
                    force.y += dx*ddy*data[iz].z*gridvalue;
                    force.z += dx*dy*ddata[iz].z*gridvalue;
                    phi += dx*dy*data[iz].z*gridvalue;
                }
            }
        }
        if (q != 0) {
            real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
            real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
            real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
            forceBuffers[atom] += (mm_ulong) realToFixedPoint(forceX);
            forceBuffers[atom+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceY);
            forceBuffers[atom+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceZ);
        }
#if NUM_DERIVS > 0
        if (b != 0)
            energyParamDerivs[GLOBAL_ID*NUM_DERIVS+derivIndex[b]] += p.x*EPSILON_FACTOR*phi;
#endif
    }
}
)MSLD";

} // namespace CommonMSLDKernelSources
} // namespace MSLDPlugin

#endif /*COMMON_MSLD_KERNEL_SOURCES_H_*/
