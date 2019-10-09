#ifndef CBLAS_ALT_TEMPLATE_H
#define CBLAS_ALT_TEMPLATE_H

#ifndef CBLAS_ALT_TEMPLATE
#define CBLAS_ALCBLAS_ALTT_TEMPLATE

//#include <cblas.h>
#include <stddef.h>
//#include <blas.h>
#ifdef small
#undef small
#endif
//#include <lapack.h>
#include <cblas_defvar.h>

#ifdef NEW_MATLAB_BLAS
#define INTT ptrdiff_t
#else
#ifdef OLD_MATLAB_BLAS
#define INTT int
#else
#ifdef MKL_INT
#define INTT MKL_INT
#else
#ifdef INT_64BITS
#define INTT long long int 
#else
#define INTT int
#endif
#endif
#endif
#endif

#ifdef INT_64BITS
#define INTM INTT
#else
#define INTM int
#endif


/// a few static variables for lapack
static char lower='L';
static char upper='u';
static INTT info=0;
static char incr='I';
static char decr='D';
static char no='N';
static char reduced='S';
static char allV='V';

#ifdef ADD_ATL 
#define dnrm2_ ATL_dnrm2
#define snrm2_ ATL_snrm2
#define dcopy_ ATL_dcopy
#define scopy_ ATL_scopy
#define daxpy_ ATL_daxpy
#define saxpy_ ATL_saxpy
#define daxpby_ ATL_daxpby
#define saxpby_ ATL_saxpby
#define dscal_ ATL_dscal
#define sscal_ ATL_sscal
#define dasum_ ATL_dasum
#define sasum_ ATL_sasum
#define ddot_ ATL_ddot
#define sdot_ ATL_sdot
#define dgemv_ ATL_dgemv
#define sgemv_ ATL_sgemv
#define dger_ ATL_dger
#define sger_ ATL_sger
#define dtrmv_ ATL_dtrmv
#define strmv_ ATL_strmv
#define dsyr_ ATL_dsyr
#define ssyr_ ATL_ssyr
#define dsymv_ ATL_dsymv
#define ssymv_ ATL_ssymv
#define dgemm_ ATL_dgemm
#define sgemm_ ATL_sgemm
#define dsyrk_ ATL_dsyrk
#define ssyrk_ ATL_ssyrk
#define dtrmm_ ATL_dtrmm
#define strmm_ ATL_strmm
#define dtrtri_ ATL_dtrtri
#define strtri_ ATL_strtri
#define idamax_ ATL_idamax
#define isamax_ ATL_isamax
#endif

#ifdef REMOVE_
#define dnrm2_ dnrm2
#define snrm2_ snrm2
#define dcopy_ dcopy
#define scopy_ scopy
#define daxpy_ daxpy
#define saxpy_ saxpy
#define daxpby_ daxpby
#define saxpby_ saxpby
#define dscal_ dscal
#define sscal_ sscal
#define dasum_ dasum
#define sasum_ sasum
#define ddot_ ddot
#define sdot_ sdot
#define ddoti_ ddoti
#define sdoti_ sdoti
#define dgemv_ dgemv
#define sgemv_ sgemv
#define dger_ dger
#define sger_ sger
#define dtrmv_ dtrmv
#define strmv_ strmv
#define dsyr_ dsyr
#define ssyr_ ssyr
#define dsymv_ dsymv
#define ssymv_ ssymv
#define dgemm_ dgemm
#define sgemm_ sgemm
#define dsyrk_ dsyrk
#define ssyrk_ ssyrk
#define dtrmm_ dtrmm
#define strmm_ strmm
#define dtrtri_ dtrtri
#define strtri_ strtri
#define idamax_ idamax
#define isamax_ isamax
#define dsytrf_ dsytrf
#define ssytrf_ ssytrf
#define dsytri_ dsytri
#define ssytri_ ssytri
#define dlasrt_ dlasrt
#define slasrt_ slasrt
#define dgesvd_ dgesvd
#define sgesvd_ sgesvd
#define dsyev_ dsyev
#define ssyev_ ssyev
#endif

/// external functions
//#ifdef HAVE_MKL   // obsolete
//extern "C" {
//#endif
INTT cblas_idamin( INTT n,  double* X,  INTT incX);
INTT cblas_isamin( INTT n,  float* X,  INTT incX);
//#ifdef HAVE_MKL
//};
//#endif

//#define HAVE_MKL
/*#ifdef HAVE_MKL   // obsolete, do not use

#define idamin_ idamin
#define isamin_ isamin
extern "C" {
   void vdSqr( int n,  double* vecIn, double* vecOut);
   void vsSqr( int n,  float* vecIn, float* vecOut);
   void vdSqrt( int n,  double* vecIn, double* vecOut);
   void vsSqrt( int n,  float* vecIn, float* vecOut);
   void vdInvSqrt( int n,  double* vecIn, double* vecOut);
   void vsInvSqrt( int n,  float* vecIn, float* vecOut);
   void vdSub( int n,  double* vecIn,  double* vecIn2, double* vecOut);
   void vsSub( int n,  float* vecIn,  float* vecIn2, float* vecOut);
   void vdDiv( int n,  double* vecIn,  double* vecIn2, double* vecOut);
   void vsDiv( int n,  float* vecIn,  float* vecIn2, float* vecOut);
   void vdExp( int n,  double* vecIn, double* vecOut);
   void vsExp( int n,  float* vecIn, float* vecOut);
   void vdInv( int n,  double* vecIn, double* vecOut);
   void vsInv( int n,  float* vecIn, float* vecOut);
   void vdAdd( int n,  double* vecIn,  double* vecIn2, double* vecOut);
   void vsAdd( int n,  float* vecIn,  float* vecIn2, float* vecOut);
   void vdMul( int n,  double* vecIn,  double* vecIn2, double* vecOut);
   void vsMul( int n,  float* vecIn,  float* vecIn2, float* vecOut);
   void vdAbs( int n,  double* vecIn, double* vecOut);
   void vsAbs( int n,  float* vecIn, float* vecOut);
}
#endif*/


// INTTerfaces to a few BLAS function, Level 1
/// INTTerface to cblas_*nrm2
template <typename T> T inline cblas_nrm2( INTT n,  T* X,  INTT incX);
/// INTTerface to cblas_*copy
template <typename T> void inline cblas_copy( INTT n,  T* X,  INTT incX, 
      T* Y,  INTT incY);
/// INTTerface to cblas_*axpy
template <typename T> void inline cblas_axpy( INTT n,  T a,  T* X, 
       INTT incX, T* Y,  INTT incY);
template <typename T> void inline cblas_axpby( INTT n,  T a,  T* X, 
       INTT incX, T b,  T* Y,  INTT incY);
/// INTTerface to cblas_*scal
template <typename T> void inline cblas_scal( INTT n,  T a, T* X, 
       INTT incX);
/// INTTerface to cblas_*asum
template <typename T> T inline cblas_asum( INTT n,  T* X,  INTT incX);
/// INTTerface to cblas_*adot
template <typename T> T inline cblas_dot( INTT n,  T* X,  INTT incX, 
       T* Y, INTT incY);
/// interface to cblas_i*amin
template <typename T> INTT inline cblas_iamin( INTT n,  T* X,  INTT incX);
/// interface to cblas_i*amax
template <typename T> INTT inline cblas_iamax( INTT n,  T* X,  INTT incX);

// INTTerfaces to a few BLAS function, Level 2

/// INTTerface to cblas_*gemv
template <typename T> void cblas_gemv( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M, 
       INTT N,  T alpha,  T *A,  INTT lda,  T *X, 
       INTT incX,  T beta,T *Y,   INTT incY);
/// INTTerface to cblas_*trmv
template <typename T> void inline cblas_trmv( CBLAS_ORDER order,  CBLAS_UPLO Uplo,
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,  INTT N,
       T *A,  INTT lda, T *X,  INTT incX);
/// INTTerface to cblas_*syr
template <typename T> void inline cblas_syr( CBLAS_ORDER order, 
        CBLAS_UPLO Uplo,  INTT N,  T alpha, 
       T *X,  INTT incX, T *A,  INTT lda);

/// INTTerface to cblas_*symv
template <typename T> inline void cblas_symv( CBLAS_ORDER order,
       CBLAS_UPLO Uplo,  INTT N, 
       T alpha,  T *A,  INTT lda,  T *X, 
       INTT incX,  T beta,T *Y,   INTT incY);


// INTTerfaces to a few BLAS function, Level 3
/// INTTerface to cblas_*gemm
template <typename T> void cblas_gemm( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  T alpha, 
       T *A,  INTT lda,  T *B,  INTT ldb,
       T beta, T *C,  INTT ldc);
/// INTTerface to cblas_*syrk
template <typename T> void cblas_syrk( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       T alpha,  T *A,  INTT lda,
       T beta, T*C,  INTT ldc);
/// INTTerface to cblas_*ger
template <typename T> void cblas_ger( CBLAS_ORDER order, 
       INTT M,  INTT N,  T alpha,  T *X,  INTT incX,
       T* Y,  INTT incY, T*A,  INTT lda);
/// INTTerface to cblas_*trmm
template <typename T> void cblas_trmm( CBLAS_ORDER order, 
       CBLAS_SIDE Side,  CBLAS_UPLO Uplo, 
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,
       INTT M,  INTT N,  T alpha, 
       T*A,  INTT lda,T *B,  INTT ldb);

// interfaces to a few functions from the intel Vector Mathematical Library
/// INTTerface to v*Sqr
template <typename T> void vSqrt( INTT n,  T* vecIn, T* vecOut);
/// INTTerface to v*Sqr
template <typename T> void vInvSqrt( INTT n,  T* vecIn, T* vecOut);
/// INTTerface to v*Sqr
template <typename T> void vSqr( INTT n,  T* vecIn, T* vecOut);
/// INTTerface to v*Sub
template <typename T> void vSub( INTT n,  T* vecIn,  T* vecIn2, T* vecOut);
/// INTTerface to v*Div
template <typename T> void vDiv( INTT n,  T* vecIn,  T* vecIn2, T* vecOut);
/// INTTerface to v*Exp
template <typename T> void vExp( INTT n,  T* vecIn, T* vecOut);
/// INTTerface to v*Inv
template <typename T> void vInv( INTT n,  T* vecIn, T* vecOut);
/// INTTerface to v*Add
template <typename T> void vAdd( INTT n,  T* vecIn,  T* vecIn2, T* vecOut);
/// INTTerface to v*Mul
template <typename T> void vMul( INTT n,  T* vecIn,  T* vecIn2, T* vecOut);
/// INTTerface to v*Abs
template <typename T> void vAbs( INTT n,  T* vecIn, T* vecOut);

// interfaces to a few LAPACK functions
/// interface to *trtri
template <typename T> void trtri(char& uplo, char& diag, 
      INTT n, T * a, INTT lda);
/// interface to *sytri  // call sytrf
template <typename T> void sytri(char& uplo, INTT n, T* a, INTT lda);
//, INTT* ipiv,
//      T* work);
/// interaface to *lasrt
template <typename T> void lasrt(char& id,  INTT n, T *d);
//template <typename T> void lasrt2(char& id,  INTT& n, T *d, int* key);
template <typename T> void gesvd( char& jobu, char& jobvt, INTT m, 
      INTT n, T* a, INTT lda, T* s,
      T* u, INTT ldu, T* vt, INTT ldvt);
template <typename T> void syev( char& jobz, char& uplo, INTT n,
         T* a, INTT lda, T* w);


/* ******************
 * Implementations
 * *****************/

extern "C" {
   double dnrm2_(INTT *n,double *x,INTT *incX);
   float snrm2_(INTT *n,float *x,INTT *incX);
   void dcopy_(INTT *n,double *x,INTT *incX, double *y,INTT *incY);
   void scopy_(INTT *n,float *x,INTT *incX, float *y,INTT *incY);
   void daxpy_(INTT *n,double* a, double *x,INTT *incX, double *y,INTT *incY);
   void saxpy_(INTT *n,float* a, float *x,INTT *incX, float *y,INTT *incY);
   void daxpby_(INTT *n,double* a, double *x,INTT *incX,double* b,  double *y,INTT *incY);
   void saxpby_(INTT *n,float* a, float *x,INTT *incX, float* b, float *y,INTT *incY);
   void dscal_(INTT *n,double* a, double *x,INTT *incX);
   void sscal_(INTT *n,float* a, float *x,INTT *incX);
   double dasum_(INTT *n,double *x,INTT *incX);
   float sasum_(INTT *n,float *x,INTT *incX);
   double ddot_(INTT *n,double *x,INTT *incX, double *y,INTT *incY);
   float sdot_(INTT *n,float *x,INTT *incX, float *y,INTT *incY);
   void dgemv_(char   *trans, INTT *m, INTT *n, double *alpha, double *a,
         INTT *lda, double *x, INTT *incx, double *beta, double *y,INTT *incy);
   void sgemv_(char   *trans, INTT *m, INTT *n, float *alpha, float *a,
         INTT *lda, float *x, INTT *incx, float *beta, float *y,INTT *incy);
   void dger_(INTT *m, INTT *n, double *alpha, double *x, INTT *incx,
         double *y, INTT *incy, double *a, INTT *lda);
   void sger_(INTT *m, INTT *n, float *alpha, float *x, INTT *incx,
         float *y, INTT *incy, float *a, INTT *lda);
   void dtrmv_(char *uplo, char *trans, char   *diag, INTT *n, double *a,
         INTT *lda, double *x, INTT *incx);
   void strmv_(char *uplo, char *trans, char   *diag, INTT *n, float *a,
         INTT *lda, float *x, INTT *incx);
   void dsyr_(char *uplo, INTT *n, double *alpha, double *x, INTT *incx,
         double *a, INTT *lda);
   void ssyr_(char *uplo, INTT *n, float *alpha, float *x, INTT *incx,
         float *a, INTT *lda);
   void dsymv_(char   *uplo, INTT *n, double *alpha, double *a, INTT *lda,
         double *x, INTT *incx, double *beta, double *y, INTT *incy);
   void ssymv_(char   *uplo, INTT *n, float *alpha, float *a, INTT *lda,
         float *x, INTT *incx, float *beta, float *y, INTT *incy);
   void dgemm_(char *transa, char *transb, INTT *m, INTT *n, INTT *k,
         double *alpha, double *a, INTT *lda, double *b, INTT *ldb, double *beta,
         double *c, INTT *ldc);
   void sgemm_(char *transa, char *transb, INTT *m, INTT *n, INTT *k,
         float *alpha, float *a, INTT *lda, float *b, INTT *ldb, float *beta,
         float *c, INTT *ldc);
   void dsyrk_(char *uplo, char *trans, INTT *n, INTT *k, double *alpha,
         double *a, INTT *lda, double *beta, double *c, INTT *ldc);
   void ssyrk_(char *uplo, char *trans, INTT *n, INTT *k, float *alpha,
         float *a, INTT *lda, float *beta, float *c, INTT *ldc);
   void dtrmm_(char *side,char *uplo,char *transa, char *diag, INTT *m,
         INTT *n, double *alpha, double *a, INTT *lda, double *b, 
         INTT *ldb);
   void strmm_(char *side,char *uplo,char *transa, char *diag, INTT *m,
         INTT *n, float *alpha, float *a, INTT *lda, float *b, 
         INTT *ldb);
   INTT idamax_(INTT *n, double *dx, INTT *incx);
   INTT isamax_(INTT *n, float *dx, INTT *incx);
   void dtrtri_(char* uplo, char* diag, INTT* n, double * a, INTT* lda, 
         INTT* info);
   void strtri_(char* uplo, char* diag, INTT* n, float * a, INTT* lda, 
         INTT* info);
   void dsytrf_(char* uplo, INTT* n, double* a, INTT* lda, INTT* ipiv,
         double* work, INTT* lwork, INTT* info);
   void ssytrf_(char* uplo, INTT* n, float* a, INTT* lda, INTT* ipiv,
         float* work, INTT* lwork, INTT* info);
   void dsytri_(char* uplo, INTT* n, double* a, INTT* lda, INTT* ipiv,
         double* work, INTT* info);
   void ssytri_(char* uplo, INTT* n, float* a, INTT* lda, INTT* ipiv,
         float* work, INTT* info);
   void dlasrt_(char* id,  INTT* n, double *d, INTT* info);
   void slasrt_(char* id,  INTT* n, float*d, INTT* info);
   void dgesvd_(char*jobu, char *jobvt, INTT *m, INTT *n, double *a,
         INTT *lda, double *s, double *u, INTT *ldu, double *vt,
         INTT *ldvt, double *work, INTT *lwork, INTT *info);
   void sgesvd_(char*jobu, char *jobvt, INTT *m, INTT *n, float *a,
         INTT *lda, float *s, float *u, INTT *ldu, float *vt,
         INTT *ldvt, float *work, INTT *lwork, INTT *info);
   void dsyev_(char *jobz, char *uplo, INTT *n, double *a, INTT *lda,
         double *w, double *work, INTT *lwork, INTT *info);
   void ssyev_(char *jobz, char *uplo, INTT *n, float *a, INTT *lda,
         float *w, float *work, INTT *lwork, INTT *info);
}

// Implementations of the INTTerfaces, BLAS Level 1
/// Implementation of the INTTerface for cblas_dnrm2
template <> inline double cblas_nrm2<double>( INTT n,  double* X, 
       INTT incX) {
   //return cblas_dnrm2(n,X,incX);
   return dnrm2_(&n,X,&incX);
};
/// Implementation of the INTTerface for cblas_snrm2
template <> inline float cblas_nrm2<float>( INTT n,  float* X, 
       INTT incX) {
   //return cblas_snrm2(n,X,incX);
   return snrm2_(&n,X,&incX);
};
/// Implementation of the INTTerface for cblas_dcopy
template <> inline void cblas_copy<double>( INTT n,  double* X, 
       INTT incX, double* Y,  INTT incY) {
   //cblas_dcopy(n,X,incX,Y,incY);
   dcopy_(&n,X,&incX,Y,&incY);
};
/// Implementation of the INTTerface for cblas_scopy
template <> inline void cblas_copy<float>( INTT n,  float* X,  INTT incX, 
      float* Y,  INTT incY) {
   //cblas_scopy(n,X,incX,Y,incY);
   scopy_(&n,X,&incX,Y,&incY);
};
/// Implementation of the INTTerface for cblas_scopy
template <> inline void cblas_copy<INTM>( INTT n, INTM* X,  INTT incX, 
      INTM* Y,  INTT incY) {
   for (int i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};
/// Implementation of the INTTerface for cblas_scopy
/*template <> inline void cblas_copy<long long int>( INTT n,  long long int* X,  INTT incX, 
      long long int* Y,  INTT incY) {
   for (INTT i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};*/

/// Implementation of the INTTerface for cblas_scopy
template <> inline void cblas_copy<bool>( INTT n,  bool* X,  INTT incX, 
      bool* Y,  INTT incY) {
   for (int i = 0; i<n; ++i)
      Y[incY*i]=X[incX*i];
};

/// Implementation of the INTTerface for cblas_daxpy
template <> inline void cblas_axpy<double>( INTT n,  double a,  double* X, 
       INTT incX, double* Y,  INTT incY) {
   //cblas_daxpy(n,a,X,incX,Y,incY);
   daxpy_(&n,&a,X,&incX,Y,&incY);
};
/// Implementation of the INTTerface for cblas_saxpy
template <> inline void cblas_axpy<float>( INTT n,  float a,  float* X,
       INTT incX, float* Y,  INTT incY) {
   //cblas_saxpy(n,a,X,incX,Y,incY);
   saxpy_(&n,&a,X,&incX,Y,&incY);
};

#ifndef AXPBY
template <> inline void cblas_axpby<double>( INTT n,  double a,  double* X, 
       INTT incX, double b, double* Y,  INTT incY) {
   dscal_(&n,&b,Y,&incY);
   daxpy_(&n,&a,X,&incX,Y,&incY);
};
/// Implementation of the INTTerface for cblas_saxpy
template <> inline void cblas_axpby<float>( INTT n,  float a,  float* X,
       INTT incX, float b, float* Y,  INTT incY) {
   sscal_(&n,&b,Y,&incY);
   saxpy_(&n,&a,X,&incX,Y,&incY);
};
#else
template <> inline void cblas_axpby<double>( INTT n,  double a,  double* X, 
       INTT incX, double b, double* Y,  INTT incY) {
   daxpby_(&n,&a,X,&incX,&b,Y,&incY);
};
/// Implementation of the INTTerface for cblas_saxpy
template <> inline void cblas_axpby<float>( INTT n,  float a,  float* X,
       INTT incX, float b, float* Y,  INTT incY) {
   saxpby_(&n,&a,X,&incX,&b,Y,&incY);
};
#endif

/// Implementation of the INTTerface for cblas_saxpy
template <> inline void cblas_axpy<INTM>( INTT n,  INTM a, INTM* X,
       INTT incX, INTM* Y,  INTT incY) {
   for (INTT i = 0; i<n; ++i)
      Y[i] += a*X[i];
};

/// Implementation of the INTTerface for cblas_saxpy
/*template <> inline void cblas_axpy<long long int>( INTT n,  long long int a,   long long int* X,
       INTT incX,  long long int* Y,  INTT incY) {
   for (INTT i = 0; i<n; ++i)
      Y[i] += a*X[i];
};*/


/// Implementation of the INTTerface for cblas_saxpy
template <> inline void cblas_axpy<bool>( INTT n,  bool a,  bool* X,
       INTT incX, bool* Y,  INTT incY) {
   for (int i = 0; i<n; ++i)
      Y[i] = a & X[i];
};


/// Implementation of the INTTerface for cblas_dscal
template <> inline void cblas_scal<double>( INTT n,  double a, double* X,
       INTT incX) {
   //cblas_dscal(n,a,X,incX);
   dscal_(&n,&a,X,&incX);
};
/// Implementation of the INTTerface for cblas_sscal
template <> inline void cblas_scal<float>( INTT n,  float a, float* X, 
       INTT incX) {
   //cblas_sscal(n,a,X,incX);
   sscal_(&n,&a,X,&incX);
};
/// Implementation of the INTTerface for cblas_sscal
template <> inline void cblas_scal<INTM>( INTT n,  INTM a, INTM* X, 
       INTT incX) {
   for (INTT i = 0; i<n; ++i) X[i*incX]*=a;
};
/*template <> inline void cblas_scal<long long int>( INTT n,  long long int a, long long int* X, INTT incX) {
   for (long long int i = 0; i < n; ++i) X[i*incX]*=a;
};*/
/// Implementation of the INTTerface for cblas_sscal
template <> inline void cblas_scal<bool>( INTT n,  bool a, bool* X, 
       INTT incX) {
   /// not implemented
};

/// Implementation of the INTTerface for cblas_dasum
template <> inline double cblas_asum<double>( INTT n,  double* X,  INTT incX) {
   //return cblas_dasum(n,X,1);
   return dasum_(&n,X,&incX);
};
/// Implementation of the INTTerface for cblas_sasum
template <> inline float cblas_asum<float>( INTT n,  float* X,  INTT incX) {
   //return cblas_sasum(n,X,1);
   return sasum_(&n,X,&incX);
};
/// Implementation of the INTTerface for cblas_ddot
template <> inline double cblas_dot<double>( INTT n,  double* X,
       INTT incX,  double* Y, INTT incY) {
   //return cblas_ddot(n,X,incX,Y,incY);
   return ddot_(&n,X,&incX,Y,&incY);
};
/// Implementation of the INTTerface for cblas_sdot
template <> inline float cblas_dot<float>( INTT n,  float* X,
       INTT incX,  float* Y, INTT incY) {
   //return cblas_sdot(n,X,incX,Y,incY);
   return sdot_(&n,X,&incX,Y,&incY);
};
template <> inline INTM cblas_dot<INTM>( INTT n, INTM* X,
       INTT incX, INTM* Y, INTT incY) {
   INTM total=0;
   INTT i,j;
   j=0;
   for (i = 0; i<n; ++i) {
      total+=X[i*incX]*Y[j];
      j+=incY;
   }
   return total;
};
/*template <> inline long long int cblas_dot<long long int>( INTT n,  long long int* X,
       INTT incX,  long long int* Y, INTT incY) {
   long long int total=0;
   INTT i,j;
   j=0;
   for (i = 0; i<n; ++i) {
      total+=X[i*incX]*Y[j];
      //j+=incY;
      j+=incY;
   }
   return total;
};*/

/// Implementation of the INTTerface for cblas_sdot
template <> inline bool cblas_dot<bool>( INTT n,  bool* X,
       INTT incX,  bool* Y, INTT incY) {
   /// not implemented
   return true;
};

#ifdef SPARSE_BLAS
/*extern "C" {
   double ddoti_(INTT *n,double *x,INTT *incX, double *y);
   float sdoti_(INTT *n,float *x,INTT *incX, float *y);
}

template <typename T> inline T cblas_doti( INTT n,T* X,  INTT* r, T* Y);
template <> inline double cblas_doti<double>( INTT n, double* X, INTT* r, 
      double* Y) {
   return ddoti_(&n,X,r,Y+1);
};
template <> inline float cblas_doti<float>( INTT n, float* X,  INTT* r, 
      float* Y) {
   return sdoti_(&n,X,r,Y+1);
};*/
// bug with the fortran-like indices
#else
template <typename T> inline T cblas_doti( INTT n, T* X,  INTT* r, 
      T* Y) {
   T sum=0;
   for (INTT i = 0; i<n; ++i) {
      sum += Y[r[i]]*X[i];
   }
   return sum;
};
#endif


// Implementations of the INTTerfaces, BLAS Level 2
///  Implementation of the INTTerface for cblas_dgemv
template <> inline void cblas_gemv<double>( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M,  INTT N,
       double alpha,  double *A,  INTT lda,
       double *X,  INTT incX,  double beta,
      double *Y,  INTT incY) {
   //cblas_dgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
   dgemv_(cblas_transpose(TransA),&M,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
};
///  Implementation of the INTTerface for cblas_sgemv
template <> inline void cblas_gemv<float>( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M,  INTT N,
       float alpha,  float *A,  INTT lda,
       float *X,  INTT incX,  float beta,
      float *Y,  INTT incY) {
   //cblas_sgemv(order,TransA,M,N,alpha,A,lda,X,incX,beta,Y,incY);
   sgemv_(cblas_transpose(TransA),&M,&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
};
///  Implementation of the INTTerface for cblas_sgemv
template <> inline void cblas_gemv<INTM>( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M,  INTT N,
       INTM alpha,  INTM*A,  INTT lda,
       INTM *X,  INTT incX,  INTM beta,
      INTM *Y,  INTT incY) {
   ///  not implemented
};
///  Implementation of the INTTerface for cblas_sgemv
template <> inline void cblas_gemv<bool>( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M,  INTT N,
       bool alpha,  bool *A,  INTT lda,
       bool *X,  INTT incX,  bool beta,
      bool *Y,  INTT incY) {
   /// not implemented
};
/*template <> inline void cblas_gemv<long long int>( CBLAS_ORDER order,
       CBLAS_TRANSPOSE TransA,  INTT M,  INTT N,
       long long int alpha,  long long int *A,  INTT lda,
       long long int *X,  INTT incX,  long long int beta,
      long long int *Y,  INTT incY) {
   ///  not implemented
};*/

///  Implementation of the INTTerface for cblas_dger
template <> inline void cblas_ger<double>( CBLAS_ORDER order, 
       INTT M,  INTT N,  double alpha,  double *X,  INTT incX,
       double* Y,  INTT incY, double *A,  INTT lda) {
   //cblas_dger(order,M,N,alpha,X,incX,Y,incY,A,lda);
   dger_(&M,&N,&alpha,X,&incX,Y,&incY,A,&lda);
};
///  Implementation of the INTTerface for cblas_sger
template <> inline void cblas_ger<float>( CBLAS_ORDER order, 
       INTT M,  INTT N,  float alpha,  float *X,  INTT incX,
       float* Y,  INTT incY, float *A,  INTT lda) {
   //cblas_sger(order,M,N,alpha,X,incX,Y,incY,A,lda);
   sger_(&M,&N,&alpha,X,&incX,Y,&incY,A,&lda);
};
///  Implementation of the INTTerface for cblas_dtrmv
template <> inline void cblas_trmv<double>( CBLAS_ORDER order,  CBLAS_UPLO Uplo,
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,  INTT N,
       double *A,  INTT lda, double *X,  INTT incX) {
   //cblas_dtrmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
   dtrmv_(cblas_uplo(Uplo),cblas_transpose(TransA),cblas_diag(Diag),&N,A,&lda,X,&incX);
};
///  Implementation of the INTTerface for cblas_strmv
template <> inline void cblas_trmv<float>( CBLAS_ORDER order,  CBLAS_UPLO Uplo,
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,  INTT N,
       float *A,  INTT lda, float *X,  INTT incX) {
   //cblas_strmv(order,Uplo,TransA,Diag,N,A,lda,X,incX);
   strmv_(cblas_uplo(Uplo),cblas_transpose(TransA),cblas_diag(Diag),&N,A,&lda,X,&incX);
};
/// Implementation of cblas_dsyr
template <> inline void cblas_syr( CBLAS_ORDER order, 
        CBLAS_UPLO Uplo,
       INTT N,  double alpha,  double*X,
       INTT incX, double *A,  INTT lda) {
   //cblas_dsyr(order,Uplo,N,alpha,X,incX,A,lda);
   dsyr_(cblas_uplo(Uplo),&N,&alpha,X,&incX,A,&lda);
};
/// Implementation of cblas_ssyr
template <> inline void cblas_syr( CBLAS_ORDER order, 
        CBLAS_UPLO Uplo,
       INTT N,  float alpha,  float*X,
       INTT incX, float *A,  INTT lda) {
   //cblas_ssyr(order,Uplo,N,alpha,X,incX,A,lda);
   ssyr_(cblas_uplo(Uplo),&N,&alpha,X,&incX,A,&lda);
};
/// Implementation of cblas_ssymv
template <> inline void cblas_symv( CBLAS_ORDER order,
       CBLAS_UPLO Uplo,  INTT N, 
       float alpha,  float *A,  INTT lda,  float *X, 
       INTT incX,  float beta,float *Y,   INTT incY) {
   //cblas_ssymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
   ssymv_(cblas_uplo(Uplo),&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}
/// Implementation of cblas_dsymv
template <> inline void cblas_symv( CBLAS_ORDER order,
       CBLAS_UPLO Uplo,  INTT N, 
       double alpha,  double *A,  INTT lda,  double *X, 
       INTT incX,  double beta,double *Y,   INTT incY) {
   //cblas_dsymv(order,Uplo,N,alpha,A,lda,X,incX,beta,Y,incY);
   dsymv_(cblas_uplo(Uplo),&N,&alpha,A,&lda,X,&incX,&beta,Y,&incY);
}


// Implementations of the INTTerfaces, BLAS Level 3
///  Implementation of the INTTerface for cblas_dgemm
template <> inline void cblas_gemm<double>( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  double alpha, 
       double *A,  INTT lda,  double *B,  INTT ldb,
       double beta, double *C,  INTT ldc) {
   //cblas_dgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
   dgemm_(cblas_transpose(TransA),cblas_transpose(TransB),&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
};
///  Implementation of the INTTerface for cblas_sgemm
template <> inline void cblas_gemm<float>( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  float alpha, 
       float *A,  INTT lda,  float *B,  INTT ldb,
       float beta, float *C,  INTT ldc) {
   //cblas_sgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
   sgemm_(cblas_transpose(TransA),cblas_transpose(TransB),&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
};
template <> inline void cblas_gemm<INTM>( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  INTM alpha, 
       INTM *A,  INTT lda,  INTM *B,  INTT ldb,
       INTM beta, INTM*C,  INTT ldc) {
   /// not implemented
};
/*template <> inline void cblas_gemm<long long int>( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  long long int alpha, 
       long long int *A,  INTT lda,  long long int *B,  INTT ldb,
       long long int beta, long long int *C,  INTT ldc) {
   /// not implemented
};*/

///  Implementation of the INTTerface for cblas_sgemm
template <> inline void cblas_gemm<bool>( CBLAS_ORDER order, 
       CBLAS_TRANSPOSE TransA,  CBLAS_TRANSPOSE TransB, 
       INTT M,  INTT N,  INTT K,  bool alpha, 
       bool *A,  INTT lda,  bool *B,  INTT ldb,
       bool beta, bool *C,  INTT ldc) {
   /// not implemented
};

///  Implementation of the INTTerface for cblas_dsyrk
template <> inline void cblas_syrk<double>( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       double alpha,  double *A,  INTT lda,
       double beta, double *C,  INTT ldc) {
   //cblas_dsyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
   dsyrk_(cblas_uplo(Uplo),cblas_transpose(Trans),&N,&K,&alpha,A,&lda,&beta,C,&ldc);
};
///  Implementation of the INTTerface for cblas_ssyrk
template <> inline void cblas_syrk<float>( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       float alpha,  float *A,  INTT lda,
       float beta, float *C,  INTT ldc) {
   //cblas_ssyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
   ssyrk_(cblas_uplo(Uplo),cblas_transpose(Trans),&N,&K,&alpha,A,&lda,&beta,C,&ldc);
};
///  Implementation of the INTTerface for cblas_ssyrk
template <> inline void cblas_syrk<INTM>( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       INTM alpha,  INTM*A,  INTT lda,
       INTM beta, INTM *C,  INTT ldc) {
   /// not implemented
};
/*template <> inline void cblas_syrk<long long int>( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       long long int alpha,  long long int *A,  INTT lda,
       long long int beta, long long int *C,  INTT ldc) {
   /// not implemented
};*/

///  Implementation of the INTTerface for cblas_ssyrk
template <> inline void cblas_syrk<bool>( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  INTT N,  INTT K,
       bool alpha,  bool *A,  INTT lda,
       bool beta, bool *C,  INTT ldc) {
   /// not implemented
};

///  Implementation of the INTTerface for cblas_dtrmm
template <> inline void cblas_trmm<double>( CBLAS_ORDER order, 
       CBLAS_SIDE Side,  CBLAS_UPLO Uplo, 
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,
       INTT M,  INTT N,  double alpha, 
       double *A,  INTT lda,double *B,  INTT ldb) {
   //cblas_dtrmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
   dtrmm_(cblas_side(Side),cblas_uplo(Uplo),cblas_transpose(TransA),cblas_diag(Diag),&M,&N,&alpha,A,&lda,B,&ldb);
};
///  Implementation of the INTTerface for cblas_strmm
template <> inline void cblas_trmm<float>( CBLAS_ORDER order, 
       CBLAS_SIDE Side,  CBLAS_UPLO Uplo, 
       CBLAS_TRANSPOSE TransA,  CBLAS_DIAG Diag,
       INTT M,  INTT N,  float alpha, 
       float *A,  INTT lda,float *B,  INTT ldb) {
   //cblas_strmm(Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb);
   strmm_(cblas_side(Side),cblas_uplo(Uplo),cblas_transpose(TransA),cblas_diag(Diag),&M,&N,&alpha,A,&lda,B,&ldb);
};
///  Implementation of the interface for cblas_idamax
template <> inline INTT cblas_iamax<double>( INTT n,  double* X,
       INTT incX) {
   //return cblas_idamax(n,X,incX);
   return static_cast<INTT >(idamax_(&n,X,&incX)-1);
};
///  Implementation of the interface for cblas_isamax
template <> inline INTT cblas_iamax<float>( INTT n,  float* X, 
       INTT incX) {
   //return cblas_isamax(n,X,incX);
   return static_cast<INTT>(isamax_(&n,X,&incX)-1);
};

// Implementations of the interfaces, LAPACK
/// Implemenation of the interface for dtrtri
template <> inline void trtri<double>(char& uplo, char& diag, 
      INTT n, double * a, INTT lda) {
   //dtrtri_(&uplo,&diag,&n,a,&lda,&info);
   dtrtri_(&uplo,&diag,&n,a,&lda,&info);
};
/// Implemenation of the interface for strtri
template <> inline void trtri<float>(char& uplo, char& diag, 
      INTT n, float* a, INTT lda) {
   //strtri_(&uplo,&diag,&n,a,&lda,&info);
   strtri_(&uplo,&diag,&n,a,&lda,&info);
};

/// Implemenation of the interface for dsytri
template <> inline void sytri<double>(char& uplo, INTT n, double* a, INTT lda) {
//, INTT* ipiv, double* work) {
   //dsytri_(&uplo,&n,a,&lda,ipiv,work,&info);
   INTT lwork=-1;
   INTT* ipiv= new INTT[n];
   double* query, *work;
   query = new double[1];
   dsytrf_(&uplo,&n,a,&lda,ipiv,query,&lwork,&info);
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   work = new double[static_cast<int>(lwork)];
   dsytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
   delete[](work);
   work = new double[static_cast<int>(2*n)];
   dsytri_(&uplo,&n,a,&lda,ipiv,work,&info);
   delete[](work);
   delete[](ipiv);
};
/// Implemenation of the interface for ssytri
template <> inline void sytri<float>(char& uplo, INTT n, float* a, INTT lda) {
   INTT lwork=-1;
   INTT* ipiv= new INTT[n];
   float* query, *work;
   query = new float[1];
   ssytrf_(&uplo,&n,a,&lda,ipiv,query,&lwork,&info);
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   work = new float[static_cast<int>(lwork)];
   ssytrf_(&uplo,&n,a,&lda,ipiv,work,&lwork,&info);
   delete[](work);
   work = new float[static_cast<int>(2*n)];
   ssytri_(&uplo,&n,a,&lda,ipiv,work,&info);
   delete[](work);
   delete[](ipiv);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, INTT n, double *d) {
   //dlasrt_(&id,const_cast<int*>(&n),d,&info);
   dlasrt_(&id,&n,d,&info);
};
/// interaface to *lasrt
template <> inline void lasrt(char& id, INTT n, float *d) {
   //slasrt_(&id,const_cast<int*>(&n),d,&info);
   slasrt_(&id,&n,d,&info);
};
//template <> inline void lasrt2(char& id, INTT& n, double *d,int* key) {
//   //dlasrt2_(&id,const_cast<int*>(&n),d,key,&info);
//   dlasrt2(&id,&n,d,key,&info);
//};
///// interaface to *lasrt
//template <> inline void lasrt2(char& id, INTT& n, float *d, int* key) {
//   //slasrt2_(&id,const_cast<int*>(&n),d,key,&info);
//   slasrt2(&id,&n,d,key,&info);
//};
template <> void inline gesvd( char& jobu, char& jobvt, INTT m, 
      INTT n, double* a, INTT lda, double* s,
      double* u, INTT ldu, double* vt, INTT ldvt) {
   double* query = new double[1];
   INTT lwork=-1;
   dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         query, &lwork, &info );
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   double* work = new double[static_cast<int>(lwork)];
   dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work, &lwork, &info );
   delete[](work);
}
template <> void inline gesvd( char& jobu, char& jobvt, INTT m, 
      INTT n, float* a, INTT lda, float* s,
      float* u, INTT ldu, float* vt, INTT ldvt) {
   float* query = new float[1];
   INTT lwork=-1;
   sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         query, &lwork, &info );
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   float* work = new float[static_cast<int>(lwork)];
   sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
         work, &lwork, &info );
   delete[](work);
}

template <> void inline syev( char& jobz, char& uplo, INTT n,
         float* a, INTT lda, float* w) {
   float* query = new float[1];
   INTT lwork=-1;
   ssyev_(&jobz,&uplo,&n,a,&lda,w,query,&lwork,&info);
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   float* work = new float[static_cast<int>(lwork)];
   ssyev_(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&info);
   delete[](work);
};

template <> void inline syev( char& jobz, char& uplo, INTT n,
         double* a, INTT lda, double* w) {
   double* query = new double[1];
   INTT lwork=-1;
   dsyev_(&jobz,&uplo,&n,a,&lda,w,query,&lwork,&info);
   lwork=static_cast<INTT>(*query); 
   delete[](query);
   double* work = new double[static_cast<int>(lwork)];
   dsyev_(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&info);
   delete[](work);
};




/// If the MKL is not present, a slow implementation is used instead.
/*#ifdef HAVE_MKL 

extern "C" {
   INTT idamin_(INTT *n, double *dx, INTT *incx);
   INTT isamin_(INTT *n, float *dx, INTT *incx);
};
/// Implemenation of the interface for vdSqr
template <> inline void vSqr<double>( int n,  double* vecIn, 
      double* vecOut) {
   vdSqr(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqr<float>( int n,  float* vecIn, 
      float* vecOut) {
   vsSqr(n,vecIn,vecOut);
};
template <> inline void vSqrt<double>( int n,  double* vecIn, 
      double* vecOut) {
   vdSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vSqrt<float>( int n,  float* vecIn, 
      float* vecOut) {
   vsSqrt(n,vecIn,vecOut);
};
template <> inline void vInvSqrt<double>( int n,  double* vecIn, 
      double* vecOut) {
   vdInvSqrt(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsSqr
template <> inline void vInvSqrt<float>( int n,  float* vecIn, 
      float* vecOut) {
   vsInvSqrt(n,vecIn,vecOut);
};

/// Implemenation of the interface for vdSub
template <> inline void vSub<double>( int n,  double* vecIn, 
       double* vecIn2, double* vecOut) {
   vdSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsSub
template <> inline void vSub<float>( int n,  float* vecIn, 
       float* vecIn2, float* vecOut) {
   vsSub(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdDiv
template <> inline void vDiv<double>( int n,  double* vecIn, 
       double* vecIn2, double* vecOut) {
   vdDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsDiv
template <> inline void vDiv<float>( int n,  float* vecIn, 
       float* vecIn2, float* vecOut) {
   vsDiv(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdExp
template <> inline void vExp<double>( int n,  double* vecIn, 
      double* vecOut) {
   vdExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsExp
template <> inline void vExp<float>( int n,  float* vecIn, 
      float* vecOut) {
   vsExp(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdInv
template <> inline void vInv<double>( int n,  double* vecIn, 
      double* vecOut) {
   vdInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vsInv
template <> inline void vInv<float>( int n,  float* vecIn, 
      float* vecOut) {
   vsInv(n,vecIn,vecOut);
};
/// Implemenation of the interface for vdAdd
template <> inline void vAdd<double>( int n,  double* vecIn, 
       double* vecIn2, double* vecOut) {
   vdAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsAdd
template <> inline void vAdd<float>( int n,  float* vecIn, 
       float* vecIn2, float* vecOut) {
   vsAdd(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vdMul
template <> inline void vMul<double>( int n,  double* vecIn, 
       double* vecIn2, double* vecOut) {
   vdMul(n,vecIn,vecIn2,vecOut);
};
/// Implemenation of the interface for vsMul
template <> inline void vMul<float>( int n,  float* vecIn, 
       float* vecIn2, float* vecOut) {
   vsMul(n,vecIn,vecIn2,vecOut);
};

/// interface to vdAbs
template <> inline void vAbs( int n,  double* vecIn, 
      double* vecOut) {
   vdAbs(n,vecIn,vecOut);
};
/// interface to vdAbs
template <> inline void vAbs( int n,  float* vecIn, 
      float* vecOut) {
   vsAbs(n,vecIn,vecOut);
};


/// implemenation of the interface of the non-offical blas, level 1 function 
/// cblas_idamin
template <> inline INTT cblas_iamin<double>(  INTT n,  double* X,
       INTT incX) {
   return static_cast<INTT>(idamin_(&n,X,&incX)-1);
};
/// implemenation of the interface of the non-offical blas, level 1 function 
/// cblas_isamin
template <> inline INTT cblas_iamin<float>( INTT n,  float* X, 
       INTT incX) {
   return static_cast<INTT >(isamin_(&n,X,&incX)-1);
};
/// slow alternative implementation of some MKL function

#else*/
/// Slow implementation of vdSqr and vsSqr
template <typename T> inline void vSqr( INTT n,  T* vecIn, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=vecIn[i]*vecIn[i];
};
template <typename T> inline void vSqrt( INTT n,  T* vecIn, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=sqr<T>(vecIn[i]);
};
template <typename T> inline void vInvSqrt( INTT n,  T* vecIn, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=T(1.0)/sqr<T>(vecIn[i]);
};

/// Slow implementation of vdSub and vsSub
template <typename T> inline void vSub( INTT n,  T* vecIn1, 
       T* vecIn2, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=vecIn1[i]-vecIn2[i];
};
/// Slow implementation of vdInv and vsInv
template <typename T> inline void vInv( INTT n,  T* vecIn, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=1.0/vecIn[i];
};
/// Slow implementation of vdExp and vsExp
template <typename T> inline void vExp( INTT n,  T* vecIn, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=exp(vecIn[i]);
};
/// Slow implementation of vdAdd and vsAdd
template <typename T> inline void vAdd( INTT n,  T* vecIn1, 
       T* vecIn2, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=vecIn1[i]+vecIn2[i];
};
/// Slow implementation of vdMul and vsMul
template <typename T> inline void vMul( INTT n,  T* vecIn1, 
       T* vecIn2, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=vecIn1[i]*vecIn2[i];
};
/// Slow implementation of vdDiv and vsDiv
template <typename T> inline void vDiv( INTT n,  T* vecIn1, 
       T* vecIn2, T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=vecIn1[i]/vecIn2[i];
};
/// Slow implementation of vAbs
template <typename T> inline void vAbs( INTT n,  T* vecIn, 
      T* vecOut) {
   for (INTT i = 0; i<n; ++i) vecOut[i]=abs<T>(vecIn[i]);
};

/// Slow implementation of cblas_idamin and cblas_isamin
template <typename T> INTT inline cblas_iamin(INTT n, T* X, INTT incX) {
   INTT imin=0;
   double min=fabs(X[0]);
   for (int j = 1; j<n; j+=incX) {
      double cur = fabs(X[j]);
      if (cur < min) {
         imin=j;
         min = cur;
      }
   }
   return imin;
}
//#endif

#endif 

#endif // CBLAS_ALT_TEMPLATE_H
