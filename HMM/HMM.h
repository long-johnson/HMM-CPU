#pragma once
#include <utility>
#include <iostream>
#include "svm.h"

#define A(i,j) A[i*N+j]
#define A1(i,j) A1[i*N+j]
#define A_used(i,j) A_used[i*N+j]
#define TAU(i,m) TAU[i*M+m]
#define TAU1(i,m) TAU1[i*M+m]
#define TAU_used(i,m) TAU_used[i*M+m]
#define MU(z,i,m) MU[((i)*M+m)*Z+z]
#define SIG(z1,z2,i,m) SIG[ ((i*M+m)*Z+z1)*Z+ z2]
#define Otr(k,t,z) Otr[(k*T+t)*Z+z]
#define MU1(z,i,m,n) MU1[(((n)*N+i)*M+m)*Z+z]
#define SIG1(z1,z2,i,m,n) SIG1[((((n)*N+i)*M+m)*Z+z1)*Z+z2]
#define B(i,t,k) B[((i)*T+t)*K+k]

#define c(t,k) c[(t)*K+k]
#define alf(t,i,k) alf[((t)*N+i)*K+k]
#define alf_t(t,i,k) alf_t[((t)*N+i)*K+k]
#define bet(t,i,k) bet[((t)*N+i)*K+k]
#define bet_t(t,i,k) bet_t[((t)*N+i)*K+k]
#define gam(t,i,k) gam[((t)*N+i)*K+k]
#define gamd(t,i,m,k) gamd[(((t)*N+i)*M+m)*K+k]
#define ksi(t,i,j,k) ksi[(((t)*N+i)*N+j)*K+k]



#define pi 3.1415926535897932384626433832795

// максимум итераций в алгоритме Баума-Велша
#define MAX_ITER 5

// требуемая точность в алгоритме Баума-Велша
#define EPS_BAUM 0.01

// используемая точность (float или double)
typedef float real_t;
//typedef double real_t;

///
/// Параметры масштабирования
///
struct svm_scaling_parameters
{
	real_t lower;
	real_t upper;
	real_t * feature_min;
	real_t * feature_max;
	void clear()
	{
		delete feature_max;
		delete feature_min;
	}
};

///
/// Класс: Скрытая Марковская Модель (Hidden Markov Model - HMM)
///
class HMM
{
public:
	
	int N,M,K,T,NumInit;	// параметры модели
	int Z;					// размерность наблюдений
	real_t *PI,*A,*TAU,*MU,*SIG,*MU1,*SIG1,*Otr,*A1,*PI1,*TAU1;		// массивы параметров и начальных приближений СММ, а также наблюдений
	real_t *alf,*bet,*c,*ksi,*gam,*gamd,*alf_t,*bet_t,*B;			// массивы вспомогательных переменных
	real_t *P_PI, *P_A, *P_TAU, *P_MU, *P_SIG;						// массивы производных по параметрам
	real_t *cd, *alf_t_d, *alf_s_d, *alf1_N, *a_N, *b_N, *dets;		// массивы вспомогательных производных

public:
	HMM(std::string filename);					// загрузка параметров модели из файла
	~HMM(void);
	void getObservations(std::string filename);						// считывание последовательностей наблюдений в класс
	void getObservations(std::string fname, real_t * Otr);		// считывание последовательностей наблюдений в массив Otr 
	void findModelParameters();					// нахождение параметров модели
	void classifyWithLikelihood(real_t * p);	// классификация последовательностей наблюдений по ф-ии правдоподобия
												// (p[k] - вероятностей того, что данная модель породила последовательность под номером k)
	///
	/// обучение по методу производных (пока что только две модели)
	/// @in: observations - наблюдения, K - число последовательностей, models - конкурирующие модели, numModels - число моделей
	/// @out: scalingParams - параметры, использованные при масштабировании производных перед обучением (передать пустую структуру)
	/// @return: svmTrainedModel * - обученная SVM модель		
	static svm_model * trainWithDerivatives(real_t ** observations, int K, HMM ** models, int numModels, svm_scaling_parameters & scalingParams);

	///
	/// классификация по методу производных
	/// @in: observations - наблюдения, K - число последовательностей, svmTrainedModel - обученная SVM модель, scalingParams - параметры, использованные при масштабировании производных перед обучением
	/// @out: r=results[k] - указывается индекс той модели, к которой была отнесена k-ое наблюдение, k=0..K-1, r=0..numModels-1
	///
	static void classifyWithDerivatives(real_t * observations, int K, svm_model & svm_trained_model, svm_scaling_parameters & scalingParams, int * results);

	// расчет и возврат производных для наблюдений извне
	void calcDerivatives(real_t * observations, int nOfSequences, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG);
private:
	// нахождение параметров алгоритомом Баума-Велша
	real_t calcBaumWelсh(int n);
	// вспомогательная функция, для расчетов в обучении и классификации
	void internal_calculations(int n);
	// вспомогательная функция для расчетов
	real_t g(int t,int k,int i,int m,int n);
	// нахождение вероятности генерации моделью наблюдений
	real_t calcProbability();

	
	// расчет производных для всех последовательностей
	void calc_derivatives_for_all_sequences(int K, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG);
	// расчет внутренних производных
	real_t calc_alpha_der(int k, real_t * alf1, real_t * a, real_t * b);
};

