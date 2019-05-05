// g++ openmp.cpp -I/home/ankit/Desktop/me766/project/ViennaCL-1.7.1 -lopencv_imgcodecs -lopencv_core
#define VIENNACL_WITH_OPENMP
// #define VIENNACL_WITH_CUDA
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

int main(){


  size_t count = 30;//img per person
  size_t no_of_people = 9;//img per person

  double * arr_n_cross_d;
  int total_no_points = THREADS_PER_BLOCK; //not thtead_per_block
  int num = 32256; //no_of_pixels
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

  double * avg_x;
  avg_x = (double *)malloc(sizeof(double) *num);

  std::vector<std::vector<double>> vec_X_n_d(0,std::vector<double>(num));
  for(int i=0;i<THREADS_PER_BLOCK;i++){
    vec_X_n_d.push_back(std::vector<double>(arr_n_cross_d+num*i,arr_n_cross_d+num*(i+1)));
  }

  for(int i=0;i<num;i++)avg_x[i]=0;

  for(int j=0;j<THREADS_PER_BLOCK;j++){
    for(int i=0;i<num;i++){
        avg_x[i]=avg_x[i]+vec_X_n_d[j][i];
    }
  }

  for(int i=0;i<num;i++)avg_x[i]/=THREADS_PER_BLOCK;

  for(int j=0;j<THREADS_PER_BLOCK;j++){
    for(int i=0;i<num;i++){
      vec_X_n_d[j][i]-=avg_x[i];
    }
  }

  viennacl::matrix<double> gpu_vec_n_d(THREADS_PER_BLOCK,num);
  viennacl::matrix<double> gpu_vec_d_n(num,THREADS_PER_BLOCK);

  viennacl::copy(vec_X_n_d, gpu_vec_n_d);
  gpu_vec_d_n = trans(gpu_vec_n_d);
  
  viennacl::matrix<double> gpu_v_l =    viennacl::linalg::prod( gpu_vec_n_d,gpu_vec_d_n);

  viennacl::matrix<double> eigenvector_l(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
  viennacl::vector<double> vcl_eigenvalues(THREADS_PER_BLOCK);

  viennacl::linalg::qr_method_sym(gpu_v_l,eigenvector_l,vcl_eigenvalues);

  viennacl::matrix<double> gpu_v = viennacl::linalg::prod(gpu_vec_d_n, eigenvector_l); //dxn
  viennacl::matrix<double> gpu_v_norm(num, THREADS_PER_BLOCK) ;

  std::vector<std::vector<double>> host_v_norm(num,std::vector<double> (THREADS_PER_BLOCK));
  
  viennacl::copy(gpu_v,host_v_norm);

  for(int i=0;i<THREADS_PER_BLOCK;i++){
      double sum = 0.0;
      for(int j=0;j<num;j++){sum+=host_v_norm[j][i] * host_v_norm[j][i];}
      for(int j=0;j<num;j++) {host_v_norm[j][i]/= sqrt(sum);}
  }
  
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
  cout<<"Exit Success\n";

  return EXIT_SUCCESS;
}