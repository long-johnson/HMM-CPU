#include <utility>
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include "HMM.h"
#include <windows.h>                // for Windows APIs

// ������� �������� ����� ������������ �������������������
void classClassify(cl_float * p1, cl_float * p2, cl_float &percent1, cl_float &percent2, int K)
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

	///
	/// ������������� OpenCL

	/// ������������� OpenCL
	///


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
	printf("Suck sess\n"); printf("Elapsed time = %f s.\n",elapsedTime);
	// ���������� ���������� ������ ����
	//
	

	//
	// �������������
	int K = M1.K;
	cl_float * p1_1 = new cl_float[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	cl_float * p1_2 = new cl_float[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	cl_float * p2_1 = new cl_float[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	cl_float * p2_2 = new cl_float[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getTestObserv("model1\\Otest1.txt");		// ������� 1 ���� � 1 ������
	M2.getTestObserv("model1\\Otest1.txt");		// ������� 1 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p1_1);				// ������������� ������������������� 1 ���� 1 �������
	M2.classifyObservations(p1_2);				// ������������� ������������������� 1 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getTestObserv("model1\\Otest2.txt");		// ������� 2 ���� � 1 ������
	M2.getTestObserv("model1\\Otest2.txt");		// ������� 2 ���� � 2 ������
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyObservations(p2_1);				// ������������� ������������������� 2 ���� 1 �������
	M2.classifyObservations(p2_2);				// ������������� ������������������� 2 ���� 2 �������
    QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Classification complete\nElapsed time = %f s.\n",elapsedTime);
	// �������������
	//

	/*using namespace std;
	cout << "p1_1\tp1_2" << endl;
	for(int i=0; i<K; i++)
		cout << p1_1[i] << "\t" << p1_2[i] << endl;
	cout << endl << "p2_1\tp2_2" << endl;
	for(int i=0; i<K; i++)
		cout << p2_1[i] << "\t" << p2_2[i] << endl;*/

	/// ������� �������� ����� ������������ � ������ ��� � ����
	cl_float succ1,fail1,succ2,fail2;
	classClassify(p1_1,p1_2,succ1,fail1,K);
	classClassify(p2_1,p2_2,fail2,succ2,K);
	std::fstream f;
	f.open("ClassClassify.txt",std::fstream::out);
	f<<(succ1+succ2)*0.5;
	f.close();

	return EXIT_SUCCESS;
}