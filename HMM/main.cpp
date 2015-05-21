#include <utility>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <windows.h>                // for Windows APIs
#include "HMM.h"

// подсчет процента верно распознанных последовательностей
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
	/// таймер
	LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks
    double elapsedTime;
    QueryPerformanceFrequency(&frequency);  // get ticks per second
	/// таймер

	//
	// ВЫЧИСЛЕНИЯ ПАРАМЕТРОВ МОДЕЛИ ТУТА
	// считаем обучающие последовательности и начальные приближения для проведения Баума-Велша
	HMM M1("model1\\");
	HMM M2("model2\\");
    QueryPerformanceCounter(&t1);	// start timer
	M1.findModelParameters();
	M2.findModelParameters();
    QueryPerformanceCounter(&t2);	// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Success\n"); printf("Elapsed time = %f s.\n",elapsedTime);
	// ВЫЧИСЛЕНИЯ ПАРАМЕТРОВ МОДЕЛИ ТУТА
	//
	

	//
	// КЛАССИФИКАЦИЯ
	int K = M1.K;
	real_t * p1_1 = new real_t[K]; for(int i=0; i<K; i++) p1_1[i]=0.;
	real_t * p1_2 = new real_t[K]; for(int i=0; i<K; i++) p1_2[i]=0.;
	real_t * p2_1 = new real_t[K]; for(int i=0; i<K; i++) p2_1[i]=0.;
	real_t * p2_2 = new real_t[K]; for(int i=0; i<K; i++) p2_2[i]=0.;
	M1.getObservations("model1\\Otest1.txt");		// считаем 1 тест в 1 модель
	M2.getObservations("model1\\Otest1.txt");		// считаем 1 тест в 2 модель
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyWithLikelihood(p1_1);				// классификация последовательностей 1 типа 1 моделью
	M2.classifyWithLikelihood(p1_2);				// классификация последовательностей 1 типа 2 моделью
    QueryPerformanceCounter(&t2);				// stop timer
    elapsedTime = (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	M1.getObservations("model1\\Otest2.txt");		// считаем 2 тест в 1 модель
	M2.getObservations("model1\\Otest2.txt");		// считаем 2 тест в 2 модель
	QueryPerformanceCounter(&t1);				// start timer
	M1.classifyWithLikelihood(p2_1);				// классификация последовательностей 2 типа 1 моделью
	M2.classifyWithLikelihood(p2_2);				// классификация последовательностей 2 типа 2 моделью
    QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);
	printf("Classification complete\nElapsed time = %f s.\n",elapsedTime);
	// КЛАССИФИКАЦИЯ
	//

	/// подсчет процента верно распознанных и запись его в файл
	real_t succ1,fail1,succ2,fail2;
	classClassify(p1_1,p1_2,succ1,fail1,K);
	classClassify(p2_1,p2_2,fail2,succ2,K);
	std::cout << (succ1 + succ2)/2 << std::endl;
	std::fstream f;
	f.open("ClassificationResults.txt",std::fstream::out);
	f<<(succ1+succ2)*0.5;
	f.close();




	///
	/// Обучение с помощью производных
	///
	real_t * Olearn1 = new real_t[K * M1.T * M1.Z];
	real_t * Olearn2 = new real_t[K * M2.T * M2.Z];
	real_t * trainingObservations[2] = { Olearn1, Olearn2 };	// подготовим обучающие наблюдения
	HMM * models[2] = { &M1, &M2 };						// подготовим массив моделей
	M1.getObservations("model1\\Ok.txt", Olearn1);		// read learn observations for model 1
	M2.getObservations("model2\\Ok.txt", Olearn2);		// read learn observations for model 2
	svm_scaling_parameters scalingParameters;
	QueryPerformanceCounter(&t1);				// start timer
	svm_model * trainedModel = HMM::trainWithDerivatives(trainingObservations, K, models, 2, scalingParameters);
	QueryPerformanceCounter(&t2);				// stop timer
	elapsedTime += (1.0*t2.QuadPart - 1.0*t1.QuadPart) / (frequency.QuadPart*1.0);

	printf("Derivatives learning complete\nElapsed time = %f s.\n", elapsedTime);



	///
	/// Классификация с помощью производных
	///
	// загрузим тестовые последовательности в массив Otest
	real_t * Otest = new real_t[2* K * M1.T * M1.Z];
	M1.getObservations("model1\\Otest1.txt", Otest);
	M2.getObservations("model1\\Otest2.txt", &Otest[K*M1.T*M1.Z]);
	
	// подготовим массив результатов
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