// g++ vie.cpp -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core
// #define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_CUDA
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <algorithm>
#include <iterator>
#include <bits/stdc++.h>
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/qr-method-common.hpp"
// 
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/fft_operations.hpp"
#include "viennacl/linalg/qr-method.hpp"
#include "viennacl/io/matrix_market.hpp"

#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
  
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define THREADS_PER_BLOCK 270 //total no of photos

// typedef double  double;

void my_cudasafe( cudaError_t error, char const *message)
{
  if(error!=cudaSuccess) 
  { 
    fprintf(stderr,"ERROR: %s : %s\n",message,cudaGetErrorString(error)); 
    exit(-1); 
  }
}

__global__ void cuda_mean(double *matrix_n_d, double *avg_vec_d){
 
  __shared__ double temp[THREADS_PER_BLOCK];
  int index = threadIdx.x * gridDim.x + blockIdx.x;

  temp[threadIdx.x]=matrix_n_d[index];

    __syncthreads();

  if( 0 == threadIdx.x ) {
    double sum = 0;
    for( int i = 0; i < THREADS_PER_BLOCK; i++ )
      sum += temp[i];
    avg_vec_d[blockIdx.x]=sum/THREADS_PER_BLOCK;
  }

  __syncthreads();

  matrix_n_d[index]-=avg_vec_d[blockIdx.x];
}

__global__ void matrixMulKernel(double *a , double *Result, int d, int n)
{ 
  int row = blockIdx.x;
  int col = threadIdx.x;
  double tmpSum = 0;
  if (row < n && col < n) {
      for (int i = 0; i < d; i++) {
          tmpSum += a[row * d + i] * a[col * d + i];
      }
    Result[row * n + col] = tmpSum;
  }
}


int main(){


  size_t count = 30 ;//img per person
  size_t no_of_people = 9 ;//img per person

  double * arr_n_cross_d;
  int total_no_points = THREADS_PER_BLOCK; //not thtead_per_block
  int num = 32256;
  int size_of_data = sizeof(double) * num * total_no_points;

  arr_n_cross_d = (double *)malloc(size_of_data);

  vector<cv::Mat> images;

  int ith = 0;
  for(int people =1;people<=no_of_people;people++){
      vector<cv::String> fn;
      glob("images/yaleB0"+std::to_string(people) +"/*.pgm", fn, false);
      for (size_t i=0; i<count; i++){
        images.push_back(cv::imread(fn[i]));
        cv::Mat temp = images[i];
        temp.convertTo(temp,CV_64FC1);
        double *temp_d = (double * ) temp.data;
        std::copy(temp_d,temp_d+num, arr_n_cross_d+ ith*num);
        ith++;
    }
  }


  double * dev_arr_n_cross_d;
  cudaMalloc((void **)&dev_arr_n_cross_d, size_of_data);
  my_cudasafe(cudaMemcpy(dev_arr_n_cross_d, arr_n_cross_d, size_of_data, cudaMemcpyHostToDevice),"Cuda memcopy : full_array");

  double * avg_x;
  avg_x = (double *)malloc(sizeof(double) *num);


  double * dev_arr_d;
  cudaMalloc((void **)&dev_arr_d, sizeof(double) *num);

  cuda_mean<<<num,THREADS_PER_BLOCK>>>(dev_arr_n_cross_d,dev_arr_d);
  my_cudasafe(cudaGetLastError(),"Kernel invocation: calculate mean ");

  my_cudasafe(cudaMemcpy(arr_n_cross_d ,dev_arr_n_cross_d, size_of_data, cudaMemcpyDeviceToHost),"Cuda memcopy : X");
  my_cudasafe(cudaMemcpy(avg_x ,dev_arr_d, sizeof(double) *num, cudaMemcpyDeviceToHost),"Cuda memcopy : X-avg");

  std::vector<std::vector<double>> vec_X_n_d(0,std::vector<double>(num));
  for(int i=0;i<THREADS_PER_BLOCK;i++){
    vec_X_n_d.push_back(std::vector<double>(arr_n_cross_d+num*i,arr_n_cross_d+num*(i+1)));
  }

  viennacl::matrix<double> gpu_vec_n_d(THREADS_PER_BLOCK,num);
  viennacl::matrix<double> gpu_vec_d_n(num,THREADS_PER_BLOCK);

  viennacl::copy(vec_X_n_d, gpu_vec_n_d);
  gpu_vec_d_n = trans(gpu_vec_n_d);

  double * dev_L;
  cudaMalloc((void **)&dev_L, sizeof(double) *THREADS_PER_BLOCK*THREADS_PER_BLOCK);

  matrixMulKernel<<<THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_arr_n_cross_d,dev_L,num,THREADS_PER_BLOCK);
  my_cudasafe(cudaGetLastError(),"Kernel invocation: calculate matmul ");

  double * host_l;
  host_l = (double *)malloc(sizeof(double) * THREADS_PER_BLOCK * THREADS_PER_BLOCK);
  my_cudasafe(cudaMemcpy(host_l, dev_L, sizeof(double) * THREADS_PER_BLOCK * THREADS_PER_BLOCK, cudaMemcpyDeviceToHost),"Cuda memcopy : dev_L to host_l");

  std::vector<std::vector<double>> vec_L(0,std::vector<double>(THREADS_PER_BLOCK));
  
  for(int i=0;i<THREADS_PER_BLOCK;i++){
    vec_L.push_back(std::vector<double>(host_l+THREADS_PER_BLOCK*i, host_l+THREADS_PER_BLOCK*(i+1)));
  }

  viennacl::matrix<double> gpu_v_l(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
  viennacl::copy( vec_L,gpu_v_l);

  viennacl::matrix<double> eigenvector_l(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
  viennacl::vector<double> vcl_eigenvalues(THREADS_PER_BLOCK);

  // cout<<"eigenValue\n";
  viennacl::linalg::qr_method_sym(gpu_v_l,eigenvector_l,vcl_eigenvalues);
  // cout<<eigenvector_l;

  // cout<<gpu_vec_d_n<<"abddbdsfdsfd";
  viennacl::matrix<double> gpu_v = viennacl::linalg::prod(gpu_vec_d_n, eigenvector_l); //dxn
  // cout<<gpu_vec_d_n<<"abddbdsfdsfd";
  viennacl::matrix<double> gpu_v_norm(num, THREADS_PER_BLOCK) ;

  // cout<<"mult\n";
  std::vector<std::vector<double>> host_v_norm(num,std::vector<double> (THREADS_PER_BLOCK));
  // cout<<"mult\n";
  
  viennacl::copy(gpu_v,host_v_norm);

  // std::cout<<"norm started\n";
  
  for(int i=0;i<THREADS_PER_BLOCK;i++){
      double sum = 0.0;
      for(int j=0;j<num;j++){sum+=host_v_norm[j][i] * host_v_norm[j][i];}
      for(int j=0;j<num;j++) {host_v_norm[j][i]/= sqrt(sum);}
  }
  
    // std::cout<<"norm ended\n";
  
  viennacl::copy(host_v_norm,gpu_v_norm);  //gpu_v_norm = d*n  // d*k 

  viennacl::matrix<double>eigen_coeff =  viennacl::linalg::prod(gpu_vec_n_d , gpu_v_norm);  //n*k //n*n

  std::vector<vector<double>> host_eigen_coeff(THREADS_PER_BLOCK, std::vector<double>(THREADS_PER_BLOCK));

  viennacl::copy(eigen_coeff, host_eigen_coeff);





  std::vector<std::vector<double> >prob(0,std::vector<double>(num) );
  prob.push_back(vec_X_n_d[0]);

  viennacl::matrix<double> gpu_prob(1,num);

  viennacl::copy(prob, gpu_prob);

  viennacl::matrix<double> final_t = viennacl::linalg::prod( gpu_prob, gpu_v_norm);

  std::vector<std::vector<double> > prob_temp(1,std::vector<double>(THREADS_PER_BLOCK) );
  viennacl::copy(final_t,prob_temp);
  // cout<<prob_temp;
  // cout<<"tgt\n";
  // cout<<;
  double min_ = 1e20;
  int index = -1;

  for (int i = 0; i < THREADS_PER_BLOCK; i++){
    double sum =0.0;
    for(int j=0;j<THREADS_PER_BLOCK;j++){
      sum+=(prob_temp[0][i]-host_eigen_coeff[j][i])*(prob_temp[0][i]-host_eigen_coeff[j][i]);
    }
    if(sum<min_){
      min_ = sum;index=i;
    }
  }

  // cout<<index<<" "<<min_<<endl;
  cout<<"EXIT_SUCCESS";

  return EXIT_SUCCESS;
}