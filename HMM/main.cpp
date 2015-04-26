#include <utility>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <windows.h>                // for Windows APIs
#include "HMM.h"

// ������� �������� ����� ������������ �������������������
void classClassify(real_t * p1, real_t * p2, real_t &percent1, real_t &percent2, int K)
{
	percent1=percent2=0;
	for(int k=0;k<K;k++)
		if(p1[k]>p2[k])
			percent1++;
		else
			percent2++;
	percent1/=K;
	percent2/=K;
}

int main(void)
{
	/// ������
	LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
	/// ������

	//
	// ���������� ���������� ������ ����
	// ������� ��������� ������������������ � ��������� ����������� ��� ���������� �����-�����
	HMM M1("model1\\");
	HMM M2("model2\\");
    QueryPerformanceCounter(&t1);	// start timer
	M1.findModelParameters();
	M2.findModelParameters();
    QueryPerformanceCounter(&t2);	// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Success\n"); printf("Elapsed time = %f s.\n",elapsedTime);
	// ���������� ���������� ������ ����
	//
	

	//
	// �������������
	int K = M1.K;
	real_t * p1_1 = new real_t[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	real_t * p1_2 = new real_t[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	real_t * p2_1 = new real_t[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	real_t * p2_2 = new real_t[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getObservations("model1\\Otest1.txt");		// ������� 1 ���� � 1 ������
	M2.getObservations("model1\\Otest1.txt");		// ������� 1 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyWithLikelihood(p1_1);				// ������������� ������������������� 1 ���� 1 �������
	M2.classifyWithLikelihood(p1_2);				// ������������� ������������������� 1 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getObservations("model1\\Otest2.txt");		// ������� 2 ���� � 1 ������
	M2.getObservations("model1\\Otest2.txt");		// ������� 2 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyWithLikelihood(p2_1);				// ������������� ������������������� 2 ���� 1 �������
	M2.classifyWithLikelihood(p2_2);				// ������������� ������������������� 2 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Classification complete\nElapsed time = %f s.\n",elapsedTime);
	// �������������
	//

	/// ������� �������� ����� ������������ � ������ ��� � ����
	real_t succ1,fail1,succ2,fail2;
	classClassify(p1_1,p1_2,succ1,fail1,K);
	classClassify(p2_1,p2_2,fail2,succ2,K);
	std::cout << (succ1 + succ2)/2 << std::endl;
	std::fstream f;
	f.open("ClassificationResults.txt",std::fstream::out);
	f<<(succ1+succ2)*0.5;
	f.close();

	///
	/// �������� � ������� �����������
	///

	M1.getObservations("model1\\Ok.txt");	// read observations for model 1
	M2.getObservations("model2\\Ok.txt");	// read observations for model 2
	M1.learnWithDerivatives();				// train model 1 with derivatives
	M2.learnWithDerivatives();				// train model 2 with derivatives

	printf("Derivatives learning complete\n");

	///
	/// ������������� � ������� �����������
	///
	// �������� �������� ������������������
	real_t * Otest1 = new real_t[K * M1.T * M1.Z];
	real_t * Otest2 = new real_t[K * M2.T * M2.Z];
	M1.getObservations("model1\\Otest1.txt", Otest1);
	M2.getObservations("model1\\Otest2.txt", Otest2);
	// ���������� ������ �������
	HMM * models[2] = {&M1, &M2};
	// ���������� ������ �����������
	int * results1 = new int[K];
	int * results2 = new int[K];
	HMM::classifyWithDerivatives(Otest1, K, models, 2, results1);	// ������ �������� ������������������ 
	HMM::classifyWithDerivatives(Otest2, K, models, 2, results2);	// ������ �������� ������������������

	float percent = 0;
	for (int k = 0; k < K; k++)
	{
		if (results1[k] == 0)
			percent += 1;
		if (results1[k] == 1)
			percent += 1;
	}
	percent /= K*K;

	printf("Derivatives classification complete\nPercent = %d", percent);

	return EXIT_SUCCESS;
}