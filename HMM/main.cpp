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
	real_t * Olearn1 = new real_t[K * M1.T * M1.Z];
	real_t * Olearn2 = new real_t[K * M2.T * M2.Z];
	real_t * trainingObservations[2] = { Olearn1, Olearn2 };	// ���������� ��������� ����������
	HMM * models[2] = { &M1, &M2 };						// ���������� ������ �������
	M1.getObservations("model1\\Ok.txt", Olearn1);		// read learn observations for model 1
	M2.getObservations("model2\\Ok.txt", Olearn2);		// read learn observations for model 2
	svm_scaling_parameters scalingParameters;
	QueryPerformanceCounter(&t1);				// start timer
	svm_model * trainedModel = HMM::trainWithDerivatives(trainingObservations, K, models, 2, scalingParameters);
	QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);

	printf("Derivatives learning complete\nElapsed time = %f s.\n", elapsedTime);



	///
	/// ������������� � ������� �����������
	///
	// �������� �������� ������������������ � ������ Otest
	real_t * Otest = new real_t[2* K * M1.T * M1.Z];
	M1.getObservations("model1\\Otest1.txt", Otest);
	M2.getObservations("model1\\Otest2.txt", &Otest[K*M1.T*M1.Z]);
	
	// ���������� ������ �����������
	int * results = new int[2*K];
	//HMM::classifyWithDerivatives(Otest, 2*K, *trainedModel, scalingParameters, results); 

	float percent = 0;
	for (int k = 0; k < K; k++)
	{
		if (results[k] == 0)
			percent += 1;
		if (results[K+k] == 1)
			percent += 1;
	}
	percent /= K*2.0;

	printf("Derivatives classification complete\nPercent = %d\n", percent);

	return EXIT_SUCCESS;
}