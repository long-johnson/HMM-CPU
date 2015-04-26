#pragma once
#include <utility>
#include <iostream>

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

// �������� �������� � ��������� �����-�����
#define MAX_ITER 10

// ��������� �������� � ��������� �����-�����
#define EPS_BAUM 0.01

// ������������ �������� (float ��� double)
typedef float real_t;

///
/// �����: ������� ���������� ������ (Hidden Markov Model - HMM)
///
class HMM
{
public:
	
	int N,M,K,T,NumInit;	// ��������� ������
	int Z;					// ����������� ����������
	real_t *PI,*A,*TAU,*MU,*SIG,*MU1,*SIG1,*Otr,*A1,*PI1,*TAU1;		// ������� ���������� � ��������� ����������� ���, � ����� ����������
	real_t *alf,*bet,*c,*ksi,*gam,*gamd,*alf_t,*bet_t,*B;			// ������� ��������������� ����������
	real_t *P_PI, *P_A, *P_TAU, *P_MU, *P_SIG;						// ������� ����������� �� ����������
	real_t *cd, *alf_t_d, *alf_s_d, *alf1_N, *a_N, *b_N;			// ������� ��������������� �����������

public:
	HMM(std::string filename);					// �������� ���������� ������ �� �����
	~HMM(void);
	void getObservations(std::string filename);						// ���������� ������������������� ���������� � �����
	void getObservations(std::string fname, real_t * Otr);		// ���������� ������������������� ���������� � ������ Otr 
	void findModelParameters();					// ���������� ���������� ������
	void classifyWithLikelihood(real_t * p);	// ������������� ������������������� ���������� �� �-�� �������������
												// (p[k] - ������������ ����, ��� ������ ������ �������� ������������������ ��� ������� k)
	void learnWithDerivatives();				// �������� �� ������ �����������
	///
	/// ������������� �� ������ �����������
	/// @in: Ok - ����������, K - ����� �������������������, models - ������������� ������, numModels - ����� �������
	/// @out: r=results[k] - ����������� ������ ��� ������, � ������� ���� �������� k-�� ����������, k=0..K-1, r=0..numModels-1
	///
	static void classifyWithDerivatives(real_t * Ok, int K, HMM ** models, int numModels, int * results);

	// ������ � ������� ����������� ��� ���������� �����
	void calcDerivatives(real_t * observations, int nOfSequences, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG);
private:
	// ���������� ���������� ����������� �����-�����
	real_t calcBaumWel�h(int n);
	// ��������������� �������, ��� �������� � �������� � �������������
	void internal_calculations(int n);
	// ��������������� ������� ��� ��������
	real_t g(int t,int k,int i,int m,int n);
	// ���������� ����������� ��������� ������� ����������
	real_t calcProbability();
	// ������ �����������
	void calc_derivative(int k, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG);
	// ������ ���������� �����������
	real_t calc_alpha_der(int k, real_t * alf1, real_t * a, real_t * b);
};

