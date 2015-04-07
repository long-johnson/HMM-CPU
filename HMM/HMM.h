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

typedef float real_t;

class HMM
{
public:
	// ��������� ������
	int N,M,K,T,NumInit;
	int Z;	// ����������� ����������
	real_t *PI,*A,*TAU,*MU,*SIG,*MU1,*SIG1,*Otr,*A1,*PI1,*TAU1;
	real_t *alf,*bet,*c,*ksi,*gam,*gamd,*alf_t,*bet_t,*B;

public:
	HMM(std::string filename);					// �������� ���������� ������ �� �����
	~HMM(void);
	void findModelParameters();					// ���������� ���������� ������
	void getTestObserv(std::string filename);	// ���������� �������� ������������������� ��� �������������
	void classifyObservations(real_t * p);	// ������������� ������������������� ���������� 
												// (p[k] - ������������ ����, ��� 
												// ������ ������ �������� ������������������ ��� ������� k)
private:
	// ���������� ���������� ����������� �����-�����
	real_t calcBaumWel�h(int n);
	// ��������������� �������, ��� �������� � �������� � �������������
	void internal_calculations(int n);
	// ��������������� ������� ��� ��������
	real_t g(int t,int k,int i,int m,int n);
	// ���������� ����������� ��������� ������� ����������
	real_t calcProbability();
};

