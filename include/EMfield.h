#ifndef EMFIELD_H
#define EMFIELD_H

#include "Alloc.h"
#include "Grid.h"


/** structure with field information */
struct EMfield {
    // field arrays: 4D arrays
    
    /* Electric field defined on nodes: last index is component */
    FPfield*** Ex;
    FPfield* Ex_flat;
    FPfield*** Ey;
    FPfield* Ey_flat;
    FPfield*** Ez;
    FPfield* Ez_flat;
    /* Magnetic field defined on nodes: last index is component */
    FPfield*** Bxn;
    FPfield* Bxn_flat;
    FPfield*** Byn;
    FPfield* Byn_flat;
    FPfield*** Bzn;
    FPfield* Bzn_flat;
    
    
};

/** allocate electric and magnetic field */
void field_allocate(struct grid*, struct EMfield*);

/** deallocate electric and magnetic field */
void field_deallocate(struct grid*, struct EMfield*);

/** allocate field in cuda device */
void field_cuda_allocate(struct grid *grd, struct EMfield *dev_field, struct EMfield *field);

/** transfer field from host to device */
void field_cuda_memcpy(struct grid *grd, struct EMfield *dst, struct EMfield *src, cudaMemcpyKind kind);

/** deallocate field from cuda device */
void field_cuda_deallocate(struct EMfield *dev_field);

#endif
