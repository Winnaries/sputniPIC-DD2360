#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // E on nodes
    field->Ex  = newArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey  = newArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez  = newArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delArr3(field->Ex, grd->nxn, grd->nyn);
    delArr3(field->Ey, grd->nxn, grd->nyn);
    delArr3(field->Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    delArr3(field->Bxn, grd->nxn, grd->nyn);
    delArr3(field->Byn, grd->nxn, grd->nyn);
    delArr3(field->Bzn, grd->nxn, grd->nyn);
}

void field_cuda_allocate(struct grid* grd, struct EMfield *dev_field, struct EMfield *field)
{
    *dev_field = *field; 
    
    int nx = grd->nxn; 
    int ny = grd->nyn; 
    int nz = grd->nzn; 

    cudaMalloc(&dev_field->Ex_flat, nx * ny * nz * sizeof(FPfield));
    cudaMalloc(&dev_field->Ey_flat, nx * ny * nz * sizeof(FPfield));
    cudaMalloc(&dev_field->Ez_flat, nx * ny * nz * sizeof(FPfield));

    cudaMalloc(&dev_field->Bxn_flat, nx * ny * nz * sizeof(FPfield));
    cudaMalloc(&dev_field->Byn_flat, nx * ny * nz * sizeof(FPfield));
    cudaMalloc(&dev_field->Bzn_flat, nx * ny * nz * sizeof(FPfield));
}

void field_cuda_memcpy(struct grid *grd, struct EMfield *dst, struct EMfield *src, cudaMemcpyKind kind) 
{
    int nx = grd->nxn;
    int ny = grd->nyn;
    int nz = grd->nzn;

    cudaMemcpy(dst->Ex_flat, src->Ex_flat, nx * ny * nz * sizeof(FPfield), kind);
    cudaMemcpy(dst->Ey_flat, src->Ey_flat, nx * ny * nz * sizeof(FPfield), kind);
    cudaMemcpy(dst->Ez_flat, src->Ez_flat, nx * ny * nz * sizeof(FPfield), kind);

    cudaMemcpy(dst->Bxn_flat, src->Bxn_flat, nx * ny * nz * sizeof(FPfield), kind);
    cudaMemcpy(dst->Byn_flat, src->Byn_flat, nx * ny * nz * sizeof(FPfield), kind);
    cudaMemcpy(dst->Bzn_flat, src->Bzn_flat, nx * ny * nz * sizeof(FPfield), kind);
}

void field_cuda_deallocate(struct EMfield *dev_field) 
{
    cudaFree(dev_field->Ex_flat);
    cudaFree(dev_field->Ey_flat);
    cudaFree(dev_field->Ez_flat);
    
    cudaFree(dev_field->Bxn_flat);
    cudaFree(dev_field->Byn_flat);
    cudaFree(dev_field->Bzn_flat);
}