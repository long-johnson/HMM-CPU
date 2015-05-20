#pragma once
#include "HMM.h"
#include "fstream"

HMM::HMM(std::string filename)
{
	std::fstream f;
	f.open(filename+"Params.txt",std::fstream::in);
	f>>N>>M>>Z>>T>>K>>NumInit;
	f.close();

	//ВЫДЕЛЯЕМ ПАМЯТЬ ДЛЯ ПАРАМЕТРОВ СММ
	PI = new real_t[N];				// начальное распределение вероятностей
	A = new real_t[N*N];				// вероятности переходов
	TAU = new real_t[N*M];			
	MU = new real_t[N*M*Z];
	SIG = new real_t[N*M*Z*Z];
	alf = new real_t[T*N*K];
	bet = new real_t[T*N*K];
	c = new real_t[T*K];				// коэффициенты масштаба
	ksi = new real_t[(T-1)*N*N*K];	
	gam = new real_t[T*N*K];
	gamd = new real_t[T*N*M*K];
	alf_t = new real_t[T*N*K];
	bet_t = new real_t[T*N*K];
	B = new real_t[N*T*K];			// вероятности появления наблюдений

	//начальные приближения
	A1 = new real_t[N*N];
	TAU1 = new real_t[N*M];
	MU1 = new real_t[N*M*Z*NumInit];
	SIG1 = new real_t[N*M*Z*Z*NumInit];
	PI1 = new real_t[N];
	MU1 = new real_t[N*M*Z*NumInit];
	SIG1 = new real_t[N*M*Z*Z*NumInit];
	Otr = new real_t[T*Z*K];


	f.open(filename+"PI1.txt",std::fstream::in);
	for(int i=0;i<N;i++)
		f>>PI1[i];
	f.close();

	f.open(filename+"A1.txt",std::fstream::in);
	for(int i=0;i<N;i++)
		for (int j=0;j<N;j++)
			f>>A1(i,j);
	f.close();

	f.open(filename+"TAU1.txt",std::fstream::in);
	for(int i=0;i<N;i++)
		for (int j=0;j<M;j++)
			f>>TAU1(i,j);
	f.close();

	f.open(filename+"Ok.txt",std::fstream::in);
	for(int k=0;k<K;++k)
		for(int t=0;t<T;t++)
			for (int z=0;z<Z;z++)
				f>>Otr(k,t,z);
	f.close();

	f.open(filename+"MU1.txt",std::fstream::in);
	for (int n=0;n<NumInit;n++)
		for (int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z=0;z<Z;z++)
					f>>MU1(z,i,m,n);
	f.close();

	f.open(filename+"SIG1.txt",std::fstream::in);
	for (int n=0;n<NumInit;n++)
		for (int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z1=0;z1<Z;z1++)
					for(int z2=0;z2<Z;z2++)
						f>>SIG1(z1,z2,i,m,n);
	f.close();

	// инициализируем среду OpenCL
	//initializeOpenCL();
}


HMM::~HMM(void)
{
	delete A1;
	delete TAU1;
	delete MU1;
	delete SIG1;
	delete PI1;
	delete Otr; 
}

// вспомогательная функция
void copyArray(real_t * dest, real_t * source, int n)
{
	for(int i=0; i<n; i++)
		dest[i]=source[i];
}

void HMM::findModelParameters()
{
	using namespace std;
	// выполним Баума Велша для всех начальных приближений и выбираем лучший набор параметров
	real_t p, p0 = -1000000000000000.;
	real_t * PIw = new real_t[N];
	real_t * Aw = new real_t[N*N]; 
	real_t * TAUw = new real_t[N*M];
	real_t * MUw = new real_t[Z*N*M];
	real_t * SIGw = new real_t[Z*Z*N*M];
	// n - номер приближения
	for(int n=0; n<NumInit; n++)
	{
		p = calcBaumWelсh(n);
		
		//dbg
		//cout << "ps:" << endl;
		//cout << "p= " << p << " for n= " << n << endl;
		//cout << "endps:" << endl;
		//dbg

		if(p>p0)
		{
			copyArray(PIw,PI,N);			//PIw=PI;
			copyArray(Aw,A,N*N);			//Aw=A;
			copyArray(TAUw,TAU,N*M);		//TAUw=TAU;
			copyArray(MUw,MU,Z*N*M);		//MUw=MU;
			copyArray(SIGw,SIG,Z*Z*N*M);	//SIGw=SIG;
			p0=p;
		}
	}

	copyArray(PI,PIw,N);			//PI=PIw;
	copyArray(A,Aw,N*N);			//A=Aw;
	copyArray(TAU,TAUw,N*M);		//MU=MUw;
	copyArray(MU,MUw,Z*N*M);		//SIG=SIGw;
	copyArray(SIG,SIGw,Z*Z*N*M);	//TAU=TAUw;

	
	/*cout << "p=" << p << endl;
	cout << "pi:" << endl;
	for (int i=0; i<N; i++)
		cout << PI[i] << endl;
	cout << "A:" << endl;
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)
			cout << A(i,j) << "\t";
		cout << "\n";
	}
	cout << "TAU:" << endl;
	for (int i=0; i<N; i++){
		for (int m=0; m<M; m++)
			cout << TAU(i,m) << "\t";
		cout << "\n";
	}*/

	
	delete Aw;
	delete TAUw;
	delete PIw;
	delete MUw;
	delete SIGw;
}

void HMM::classifyWithLikelihood(real_t * p)
{
	using namespace std;
	// внутренние вычисления
	internal_calculations(-1);

	// кернел 5
	//cout << "classification probabilities:" << endl;
	for(int k=0;k<K;k++){
		for(int t=0;t<T;t++)
			p[k]-=log(c(t,k));
		//cout << p[k] << endl;
	}
}

real_t HMM::g(int t,int k,int i,int m,int n)
{
	//работаем с диагональными ковариационными матрицами
	real_t det=1.,res=0.;
	real_t tmp1,tmp2;
	if (n==-1) //работа с уже полученными параметрами модели 
	{
		for (int z=0;z<Z;z++)
		{
			tmp1=SIG(z,z,i,m);
			det*=tmp1;
			tmp2=Otr(k,t,z)-MU(z,i,m);
			res+=tmp2*tmp2/tmp1;
		}		
	}
	else
	{
		for (int z=0;z<Z;z++)
		{
			tmp1=SIG1(z,z,i,m,n);
			det*=tmp1;
			tmp2=Otr(k,t,z)-MU1(z,i,m,n);
			res+=tmp2*tmp2/tmp1;
		}
	}
	res*=-0.5;
	res= (real_t) exp(res)/sqrt((real_t)pow(2.*pi,Z)*det);
	if(!_finite(res))
	{
		res=0;
	}
	return res;

}

real_t HMM::calcBaumWelсh(int n)
{
	std::fstream f; // debug
	int T1=T-1;

	real_t * gam_sum = new real_t[N];
	real_t * gamd_sum = new real_t[N*M];
	real_t * tmp3 = new real_t[Z];
	real_t tmp2;
	real_t p_pred = 100;
	real_t p = 10;

	bool F=true;
	//vector<double> tmp3(Z);

	int iter;
	for(iter = 0; iter < MAX_ITER && abs(p - p_pred) > EPS_BAUM; iter++)
	{
		// большой блок вспомогательных вычислений
		internal_calculations(n);

		// кернел 3.1
		for(int i=0;i<N;i++)
		{
			gam_sum[i]=0.;
			for(int m=0;m<M;m++)
				gamd_sum[i*M+m]=0.;
		}
		
		// кернел 3.2
		real_t ttt=0.;
		for(int t=0; t<T1 && F; t++)
		{
			for(int k=0; k<K && F; k++)

				for(int i=0;i<N && F;i++)
				{
					gam_sum[i]+=gam(t,i,k);
					if(!_finite(gam_sum[i]))
					{
						std::cout<<"bad gam_sum\n";
						//F=false;
						break;
					}
					for(int m=0;m<M;m++)
					{
						ttt=gamd_sum[i*M+m]+gamd(t,i,m,k);
						if(_finite(ttt))
							gamd_sum[i*M+m]+=gamd(t,i,m,k);
						else
						{							
							std::cout<<"bad gamd_sum\n";
							//F=false;
							break;
						}						
					}
				}
		}

		// DEBUG - gam_sum, gamd_sum - no error
		/*std::fstream f;
		f.open("debugging_gam_sum.txt",std::fstream::out);
		for (int i=0; i<N; i++)
		f << gam_sum[i] << std::endl;
		f.close();
		f.open("debugging_gamd_sum.txt", std::fstream::out);
		for (int i=0; i<N*M; i++)
			f << gamd_sum[i] << std::endl;
		f.close();*/
		// DEBUG

		/*if(!F) 
			break;*/

		
		// кернел 3.3
		for(int i=0;i<N;i++)
		{
			PI[i]=0.;
			for(int k=0;k<K;k++)
			{
				PI[i]+=gam(0,i,k);
			}
			PI[i]/=K;
		}

		// DEBUG - PI - good
		/*f.open("debugging_PI.txt",std::fstream::out);
		for (int i=0; i<N; i++)
		f << PI[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// кернел 3.4
		for(int i=0;i<N;i++)
		{
			for(int j=0;j<N;j++)
			{
				tmp2=0.;
				for(int k=0;k<K;k++)
					for(int t=0;t<T1;t++)
						tmp2+=ksi(t,i,j,k)*c(t+1,k);
				A(i,j)=tmp2/gam_sum[i];
			}
		}

		// DEBUG - A - good
		/*f.open("debugging_A.txt", std::fstream::out);
		for (int i = 0; i<N*N; i++)
			f << A[i] << std::endl;
		f.close();*/
		// DEBUG

		// кернел 3.5
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				TAU(i,m)=gamd_sum[i*M+m]/gam_sum[i];

		// DEBUG - TAU - good
		/*f.open("debugging_TAU.txt", std::fstream::out);
		for (int i = 0; i<N*M; i++)
			f << TAU[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// кернел 3.6
		for(int i=0;i<N;i++)
		{
			for(int m=0;m<M;m++)
				for(int z=0;z<Z;z++)
				{
					MU(z,i,m)=0.;
					for(int k=0;k<K;k++)
						for(int t=0;t<T;t++)
						{
							ttt=MU(z,i,m)+gamd(t,i,m,k)*Otr(k,t,z);
							if(_finite(ttt))
								MU(z,i,m)+=gamd(t,i,m,k)*Otr(k,t,z);
							else
								std::cout<<"hahaMU!\n";
						}
					MU(z,i,m)/=gamd_sum[i*M+m];
				}
		}

		// DEBUG - MU - good
		/*f.open("debugging_MU.txt", std::fstream::out);
		for (int i = 0; i<N*M*Z; i++)
			f << MU[i] << std::endl;
		f.close();*/
		// DEBUG

		// кернел 3.7
		for(int i=0;i<N;i++)
		{
			for(int m=0;m<M;m++)
				for(int z1=0;z1<Z;z1++)
					for(int z2=0;z2<Z;z2++)
					{
						SIG(z1,z2,i,m)=0.;
						for(int k=0;k<K;k++)
							for(int t=0;t<T;t++)
							{
								for(int z3=0;z3<Z;z3++)
									tmp3[z3]=Otr(k,t,z3)-MU(z3,i,m);
								ttt=SIG(z1,z2,i,m)+gamd(t,i,m,k)*tmp3[z1]*tmp3[z2];
								if(_finite(ttt))
									SIG(z1,z2,i,m)+=gamd(t,i,m,k)*tmp3[z1]*tmp3[z2];
								else
									std::cout<<"hahaSIG\n";
							}
							SIG(z1,z2,i,m)/=gamd_sum[i*M+m];
					}
		}

		// DEBUG - SIG - good
		/*f.open("debugging_SIG.txt", std::fstream::out);
		for (int i = 0; i<N*M*Z*Z; i++)
			f << SIG[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// кернел 3.8
		for(int i=0;i<N;i++)
			PI1[i]=PI[i];
		// кернел 3.9
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++)
				A1(i,j)=A(i,j);
		// кернел 3.10
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				TAU1(i,m)=TAU(i,m);
		// кернел 3.11
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z=0;z<Z;z++)
					MU1(z,i,m,n)=MU(z,i,m);
		// кернел 3.12
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z1=0;z1<Z;z1++)
					for(int z2=0;z2<Z;z2++)
						SIG1(z1,z2,i,m,n)=SIG(z1,z2,i,m);		

		p = calcProbability();		// посчитать новую вероятность
		std::swap(p, p_pred);		// поменять новую и старую вероятности местами
		//std::cout << "diff =" << abs(p - p_pred) << std::endl;
	}
	
	//std::cout << "Baum Welsh: iters = " << iter << std::endl;

	delete gamd_sum; delete gam_sum; delete tmp3;

	return p_pred;					// вернем последнюю вероятность
}


void HMM::internal_calculations(int n)
{
	int T1=T-1;
	real_t * TAU_used, * A_used, * PI_used ,*SIG_used, *MU_used;
	if (n==-1){
		TAU_used = TAU;
		A_used = A;
		PI_used = PI;
		SIG_used = SIG;
		MU_used = MU;
	}
	else{
		TAU_used = TAU1;
		A_used = A1;
		PI_used = PI1;
		SIG_used = SIG1;
		MU_used = MU1;
	}

	// кернел 1.1 (getB)
	for (int k=0;k<K;k++)
		for(int i=0;i<N;i++)
			for(int t=0;t<T;t++)
				B(i,t,k)=0;

	// кернел 1.2 (getB)
	for (int k=0;k<K;k++)
		for(int i=0;i<N;i++)
			for(int t=0;t<T;t++)
				for(int m=0;m<M;m++)
					B(i,t,k)+=TAU_used(i,m)*g(t,k,i,m,n);
	// кернел!
	for(int k=0;k<K;k++)
		for(int i=0;i<N;i++)
			for(int t=0;t<T;t++)
			{
				if(!_finite(B(i,t,k)))
					B(i,t,k)=0;
			}

	// DEBUG
	/*std::fstream f;
	f.open("debugging_B.txt",std::fstream::out);
	for (int i=0; i<N*T*K; i++)
		f << B[i] << std::endl;
	f.close();*/
	// DEBUG

	// кернел 2.1 (set_var)
	real_t atsum=0.,P=0.;
	for(int k=0;k<K;k++)
	{
		for(int t=0;t<T;t++)
		{
			//c(t,k)=0;                !
			for(int i=0;i<N;i++)
			{
				alf(t,i,k)=0;
				bet(t,i,k)=0;
				alf_t(t,i,k)=0;
				bet_t(t,i,k)=0;
			}
		}
	}

	// кернел 2.2 (set_var)
	for(int k=0;k<K;k++)
	{
		for(int i=0;i<N;i++)
		{
			alf_t(0,i,k)=PI_used[i]*B(i,0,k);
			bet(T1,i,k)=1.;
		}
	}

	// кернел 2.3 (set_var)
	for(int t=0;t<T1;t++)
	{
		for(int k=0;k<K;k++)
		{
		
			for(int i=0;i<N;i++)
			{
				atsum=0.;
				for(int j=0;j<N;j++)
					atsum+=alf_t(t,j,k);
				alf(t,i,k)=alf_t(t,i,k)/atsum;
				if(!_finite(alf(t,i,k)))
				{
					alf(t,i,k)=1./N;
				}
			}
		}
		for(int k=0;k<K;k++)
		{
			for(int i=0;i<N;i++)
			{
				atsum=0;
				for(int j=0;j<N;j++)
					atsum+=alf(t,j,k)*A_used(j,i);
				alf_t(t+1,i,k)=B(i,t+1,k)*atsum;
			}
		}
	}
	
	// DEBUG - satisfying
	/*std::fstream f;
	f.open("debugging_alf.txt",std::fstream::out);
	for (int i=0; i<N*T*K; i++)
		f << alf[i] << std::endl;
	f.close();
	f.open("debugging_alf_t.txt",std::fstream::out);
	for (int i=0; i<N*T*K; i++)
		f << alf_t[i] << std::endl;
	f.close();*/
	// DEBUG

	// 2.3.3
	// T - T1
	for(int t=0;t<T1;t++)
	{	
		for(int k=0;k<K;k++)
		{
			for(int i=0;i<N;i++)
			{
				c(t,k)=alf(t,i,k)/alf_t(t,i,k);
				if(_finite(c(t,k)))
				{
					break;
				}
			}
			if(!_finite(c(t,k)))
			{
				/*if(n==-1)
					c(t,k)=1000;			//!!!!!
				else*/
					c(t,k)=10000000;
			}
		}
	}

	// кернел 2.4 (set_var)
	for(int k=0;k<K;k++)
	{
		P=0;
		for(int i=0;i<N;i++)
		{
			atsum=0.;
			for(int j=0;j<N;j++)
				atsum+=alf_t(T1,j,k);
			alf(T1,i,k)=alf_t(T1,i,k)/atsum;
			if(!_finite(alf(T1,i,k))){                  //
					alf(T1,i,k)=1./N;	
			}
			for(int j=0;j<N;j++){                      // !!
				c(T1,k)=alf(T1,j,k)/alf_t(T1,j,k);
				if(_finite(c(T1,k)))
					break;
			}
			if(!_finite(c(T1,k)))
				c(T1,k)=1000;							//
			P+=alf(T1,i,k);
		}
	}

	// DEBUG c - satisfying
	/*std::fstream f;
	f.open("debugging_c.txt",std::fstream::out);
	for (int i=0; i<T*K; i++)
	f << c[i] << std::endl;
	f.close();*/
	// DEBUG
	
	// DEBUG проверка
	/*if(abs(P-1.)>=0.1)
	{
		std::cout<<"error!!!!! p<>1!!!!\n";
		std::cout << std::endl;
	}*/

	///
	/// Далее вычисления только для этапа обучения
	///
	// кернел 2.5 (set_var)
	if (n != -1){

		for (int t = T1 - 1; t >= 0; t--)
		{
			for (int k = 0; k < K; k++)
			{

				for (int i = 0; i < N; i++)
					bet_t(t + 1, i, k) = c(t + 1, k)*bet(t + 1, i, k);
				for (int i = 0; i < N; i++){
					for (int j = 0; j < N; j++)
						bet(t, i, k) += A_used(i, j)*B(j, t + 1, k)*bet_t(t + 1, j, k);
					if (alf(t, i, k) == 1. / N || fabs(alf(t, i, k) - 1.) < 0.01)
						bet(t, i, k) = 1.;
				}
			}
		}

		// DEBUG bet, bet_t - satisfying
		/*std::fstream f;
		f.open("debugging_bet.txt", std::fstream::out);
		for (int i = 0; i<N*T*K; i++)
		f << bet[i] << std::endl;
		f.close();
		f.open("debugging_bet_t.txt", std::fstream::out);
		for (int i = 0; i<N*T*K; i++)
		f << bet_t[i] << std::endl;
		f.close();*/
		// DEBUG

		//проверка! (set_var)
		/*for(int k=0;k<K;k++)
		{
		for(int t=0;t<T;t++)
		{
		atsum=0;
		for (int i=0;i<N;i++)
		atsum+=alf(t,i,k)*bet(t,i,k);
		if(abs(atsum-1.)>=0.1)
		std::cout<<"error!!!!\n";

		}
		}*/



		// кернел 2.6 (set_var)
		for (int k = 0; k < K; k++)
		{
			for (int t = 0; t < T; t++)
			{
				for (int i = 0; i < N; i++)
				{
					gam(t, i, k) = alf(t, i, k)*bet(t, i, k);
					/*if(!_finite(gam(t,i,k)))	// проверка
						std::cout<<"gamma\n";*/
					atsum = 0.;
					for (int m = 0; m < M; m++)
						atsum += TAU_used(i, m)*g(t, k, i, m, n);
					for (int m = 0; m < M; m++){
						gamd(t, i, m, k) = TAU_used(i, m)*g(t, k, i, m, n)*gam(t, i, k) / atsum;
						if (!_finite(gamd(t, i, m, k)))
							gamd(t, i, m, k) = TAU_used(i, m)*gam(t, i, k);
					}
				}
			}
		}

		// DEBUG gam, gamd - satisfying
		/*std::fstream f;
		f.open("debugging_gam.txt", std::fstream::out);
		for (int i = 0; i<N*T*K; i++)
		f << gam[i] << std::endl;
		f.close();
		f.open("debugging_gamd.txt", std::fstream::out);
		for (int i = 0; i<N*M*T*K; i++)
		f << gamd[i] << std::endl;
		f.close();*/
		// DEBUG

		// ПРОВОЕРКА ?? 
		/*for(int k=0;k<K;k++)
		{
		for(int i=0;i<N;i++)
		for(int t=0;t<T;t++)
		{
		atsum=0;
		for(int m=0;m<M;m++)
		atsum+=gamd(t,i,m,k);
		if(abs(atsum-gam(t,i,k))>=0.01)
		std::cout<<"error!!!\n";
		}
		}*/

		// кернел 2.7 (set_var)
		for (int k = 0; k < K; k++)
		{
			for (int t = 0; t < T1; t++)
			{
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						ksi(t, i, j, k) = alf(t, i, k)*A_used(i, j)*B(j, t + 1, k)*bet(t + 1, j, k);
			}
		}

		// DEBUG ksi - satisfying
		/*std::fstream f;
		f.open("debugging_ksi.txt", std::fstream::out);
		for (int i = 0; i<N*N*T1*K; i++)
		f << ksi[i] << std::endl;
		f.close();*/
		// DEBUG

		//  ПРОВЕРКА ?? 
		/*for(int k=0;k<K;k++)
		{
		for(int i=0;i<N;i++)
		for(int t=0;t<T1;t++)
		{
		atsum=0;
		for(int j=0;j<N;j++)
		atsum+=ksi(t,i,j,k);
		}
		}*/
	}
}

real_t HMM::calcProbability()
{
	// кернел4: без кернела =( или 2д редукция
	real_t res=0;
	for(int k=0;k<K;k++)
		for(int t=0;t<T;t++)
			res -= log(c(t,k));
	return res;
}

void HMM::getObservations(std::string fname)
{
	std::fstream f;
	f.open(fname,std::fstream::in);
	for(int k=0; k<K; k++)
		for(int t=0; t<T; t++)
			for (int z=0; z<Z; z++)
				f>>Otr(k,t,z);
	f.close();
}

void HMM::getObservations(std::string fname, real_t * Otr)
{
	std::fstream f;
	f.open(fname, std::fstream::in);
	for (int k = 0; k<K; ++k)
		for (int t = 0; t<T; t++)
			for (int z = 0; z<Z; z++)
				f >> Otr(k, t, z);
	f.close();
}


void HMM::learnWithDerivatives()
{
	// allocate memory for derivatives
	this->P_PI = new real_t[K*N];
	this->P_A = new real_t[K*N*N];
	this->P_TAU = new real_t[K*N*M];
	this->P_MU = new real_t[K*Z*N*M];
	this->P_SIG = new real_t[K*Z*N*M];
	// allocate memory for auxilary variables
	this->alf1_N = new real_t[N];
	this->a_N = new real_t[N*N];
	this->b_N = new real_t[N*T];
	this->cd = new real_t[T];
	this->alf_t_d = new real_t[T*N];
	this->alf_s_d = new real_t[T*N];
	this->dets = new real_t[N*M];

	//clear allocated memory
	/*for (int i = 0; i < N; i++) alf1_N[i] = 0;
	for (int i = 0; i < N*N; i++) a_N[i] = 0;
	for (int i = 0; i < N*T; i++) b_N[i] = 0;
	for (int i = 0; i < T; i++) cd[i] = 0;
	for (int i = 0; i < T*N; i++) alf_t_d[i] = 0;
	for (int i = 0; i < T*N; i++) alf_s_d[i] = 0;*/
	
	// carry out some internal calculations 
	internal_calculations(-1);
	
	// calc derivative for each parameter and each sequence
	for (int k = 0; k<K; k++)
		calc_derivative(k, P_PI, P_A, P_TAU, P_MU, P_SIG);

	delete alf1_N;
	delete a_N;
	delete b_N;
	delete cd;
	delete alf_t_d;
	delete alf_s_d;
	delete dets;
}

void HMM::calc_derivative(int k, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG)
{

#define d_A(k,i,j) d_A[((k)*N+i)*N+j]
#define d_TAU(k,i,m) d_TAU[((k)*N+i)*M+m]
#define d_MU(k,z,i,m) d_MU[(((k)*N+i)*M+m)*Z+z]
#define d_SIG(k,z,i,m) d_SIG[(((k)*N+i)*M+m)*Z+z]

	//clear allocated memory
	//for (int i = 0; i < N; i++) alf1_N[i] = 0;
	for (int i = 0; i < N*N; i++) a_N[i] = 0;
	for (int i = 0; i < N*T; i++) b_N[i] = 0;

	//производные по PI
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<N; j++)
			alf1_N[j] = (j == i) ? B(i, 0, k) : 0;	//ALF1_PI(j,i);
		//a=0;b=0;		
		real_t temp = calc_alpha_der(k, alf1_N, a_N, b_N);
		d_PI[k*N+i] = isfinite(temp) ? temp : 0.0;
	}

	//clear allocated memory
	for (int i = 0; i < N; i++) alf1_N[i] = 0;
	//for (int i = 0; i < N*N; i++) a_N[i] = 0;
	for (int i = 0; i < N*T; i++) b_N[i] = 0;

	//производные по A
	for (int i = 0; i<N; i++)
		for (int j = 0; j<N; j++)
		{
			for (int i1 = 0; i1<N; i1++)
				for (int j1 = 0; j1<N; j1++)
					a_N[i1*N + j1] = (i1 == i && j1 == j) ? 1 : 0;
			real_t temp = calc_alpha_der(k, alf1_N, a_N, b_N);
			d_A(k, i, j) = isfinite(temp) ? temp : 0.0;
		}


	//clear allocated memory
	//for (int i = 0; i < N; i++) alf1_N[i] = 0;
	for (int i = 0; i < N*N; i++) a_N[i] = 0;
	//for (int i = 0; i < N*T; i++) b_N[i] = 0;

	//производные по MU и SIG
	for (int i = 0; i<N; i++)
		for (int m = 0; m<M; m++)
		{
			dets[i*M + m] = 1;
			for (int z = 0; z<Z; z++)
				dets[i*M + m] *= SIG(z, z, i, m);
		}

	for (int i = 0; i<N; i++)
		for (int m = 0; m<M; m++)
			for (int z = 0; z<Z; z++)
			{
				for (int i1 = 0; i1<N; i1++)
				{
					for (int t = 0; t<T; t++)
						b_N[i1*T + t] = (i1 == i) ? 0.5 * TAU(i, m) * g(t, k, i, m, -1) * (Otr(k, t, z) - MU(z, i, m)) / SIG(z, z, i, m) : 0;
					alf1_N[i1] = PI[i] * b_N[i1*T + 0];
				}
				real_t temp = calc_alpha_der(k, alf1_N, a_N, b_N);
				d_MU(k, z, i, m) = isfinite(temp) ? temp : 0.0;

				for (int i1 = 0; i1<N; i1++)
				{
					for (int t = 0; t<T; t++)
						b_N[i1*T + t] = (i1 == i) ? TAU(i, m)*g(t, k, i, m, -1)*0.5* (pow((Otr(k, t, z) - MU(z, i, m)) / SIG(z, z, i, m), 2.0) - 1. / dets[i*M + m]) : 0;
					alf1_N[i1] = PI[i] * b_N[i1*T + 0];
				}
				temp = calc_alpha_der(k, alf1_N, a_N, b_N);
				d_SIG(k, z, i, m) = isfinite(temp) ? temp : 0.0;
			}

	//clear allocated memory
	//for (int i = 0; i < N; i++) alf1_N[i] = 0;
	//for (int i = 0; i < N*N; i++) a_N[i] = 0;
	//for (int i = 0; i < N*T; i++) b_N[i] = 0;

	//производные по TAU		
	for (int i = 0; i<N; i++)
		for (int m = 0; m<M; m++)
		{
			for (int i1 = 0; i1<N; i1++)
			{
				for (int t = 0; t<T; t++)									// fixed!
					b_N[i1*T + t] = (i1 == i) ? g(t, k, i, m, -1) : 0;		// fixed!
				alf1_N[i1] = PI[i1] * ((i1 == i) ? g(0, k, i, m, -1) : 0);	//b[i1][0];
			}
			real_t temp = calc_alpha_der(k, alf1_N, a_N, b_N);
			d_TAU(k, i, m) = isfinite(temp) ? temp : 0.0;
		}

	
}

real_t HMM::calc_alpha_der(int k, real_t * alf1_N, real_t * a_N, real_t * b_N)
{
	for (int i = 0; i < T; i++) cd[i] = 0;
	for (int i = 0; i < T*N; i++) alf_t_d[i] = 0;
	for (int i = 0; i < T*N; i++) alf_s_d[i] = 0;


	for (int i = 0; i<N; i++)
	{
		alf_t_d[0*N+i] = alf1_N[i];
		cd[0] += alf_t_d[0*N+i];
	}
	cd[0] = -c(0, k)*c(0, k)*cd[0];
	double sum1, sum2;
	for (int t = 1; t<T; t++)
	{
		for (int i = 0; i<N; i++)
			alf_s_d[(t - 1)*N+i] = cd[t - 1] * alf_t(t - 1, i, k) + alf_t_d[(t - 1)*N+i] * c(t - 1, k);
		for (int i = 0; i<N; i++)
		{
			sum1 = sum2 = 0;
			for (int j = 0; j<N; j++)
			{
				sum1 += alf_s_d[(t - 1)*N+i] * A(j, i) + alf(t - 1, j, k)*a_N[j*N+i];
				sum2 += alf(t - 1, j, k)*A(j, i);
			}
			alf_t_d[t*N+i] = sum1*B(i, t, k) + sum2*b_N[i*T+t];
			cd[t] += alf_t_d[t*N+i];
		}
		cd[t] = -c(t, k)*c(t, k)*cd[t];
	}
	double deriv = 0;
	for (int t = 0; t<T; t++)
		deriv += cd[t] / c(t, k);
	return deriv;

}

void HMM::classifyWithDerivatives(real_t * Ok, int K, HMM ** models, int numModels, int * results)
{
	// by this moment we shall compare only two models
	numModels = 2;

	int N = models[0]->N;
	int M = models[0]->M;
	int Z = models[0]->Z;

	// allocate memory for derivatives for 1st model
	real_t * d_PI_0 = new real_t[K*N];
	real_t * d_A_0 = new real_t[K*N*N];
	real_t * d_TAU_0 = new real_t[K*N*M];
	real_t * d_MU_0 = new real_t[K*Z*N*M];
	real_t * d_SIG_0 = new real_t[K*Z*Z*N*M];

	// calc derivatives for 1st model
	models[0]->calcDerivatives(Ok, K, d_PI_0, d_A_0, d_TAU_0, d_MU_0, d_SIG_0);

	// allocate memory for derivatives for 2nd model
	real_t * d_PI_1 = new real_t[K*N];
	real_t * d_A_1 = new real_t[K*N*N];
	real_t * d_TAU_1 = new real_t[K*N*M];
	real_t * d_MU_1 = new real_t[K*Z*N*M];
	real_t * d_SIG_1 = new real_t[K*Z*Z*N*M];

	// calc derivatives for 1st model
	models[1]->calcDerivatives(Ok, K, d_PI_1, d_A_1, d_TAU_1, d_MU_1, d_SIG_1);

	// prepare SVM data
	svm_problem prob;
	svm_parameter param;

	int derivativesVectorSize = N + N*N + N*M + Z*N*M + Z*Z*N*M;	// number of derivatives for each sequence
	prob.l = 2 * K;					// number of derivative vectors (for both models)
	prob.y = new double[2 * K];		// belonging of each derivative vector
	for (int i = 0; i < K; i++)
	{
		prob.y[i] = -1;
		prob.y[i + K] = 1;
	}
	prob.x = new svm_node* [2 * K];		// vectors of derivatives
	// first model
	for (int k = 0; k < K; k++)			// k - индекс последовательности наблюдений
	{
		prob.x[k] = new svm_node[derivativesVectorSize];	// allocate memeory for derivatives vector
		int j = 0;		// указатель на положение в одном векторе производных
		for (int i = 0; i < N; i++, j++)
		{
			prob.x[k][j].index =  j+1;
			prob.x[k][j].value = models[0]->P_PI[k*N + i];
		}
		for (int i = 0; i < N*N; i++, j++)
		{
			prob.x[k][j].index = j+1;
			prob.x[k][j].value = models[0]->P_A[k*N*N + i];
		}
		for (int i = 0; i < N*M; i++, j++)
		{
			prob.x[k][j].index = j+1;
			prob.x[k][j].value = models[0]->P_TAU[k*N*M + i];
		}
		for (int i = 0; i < Z*N*M; i++, j++)
		{
			prob.x[k][j].index = j+1;
			prob.x[k][j].value = models[0]->P_MU[k*Z*N*M + i];
		}
		for (int i = 0; i < Z*Z*N*M; i++, j++)
		{
			prob.x[k][j].index = j+1;
			prob.x[k][j].value = models[0]->P_SIG[k*Z*Z*N*M + i];
		}
	}
	// second model
	for (int k = 0; k < K; k++)			// k - индекс последовательности наблюдений
	{
		prob.x[K+k] = new svm_node[derivativesVectorSize];	// allocate memory for derivatives vector
		int j = 0;		// указатель на положение в одном векторе производных
		for (int i = 0; i < N; i++, j++)
		{
			prob.x[K+k][j].index = j + 1;
			prob.x[K+k][j].value = models[1]->P_PI[k*N + i];
		}
		for (int i = 0; i < N*N; i++, j++)
		{
			prob.x[K+k][j].index = j + 1;
			prob.x[K+k][j].value = models[1]->P_A[k*N*N + i];
		}
		for (int i = 0; i < N*M; i++, j++)
		{
			prob.x[K+k][j].index = j + 1;
			prob.x[K+k][j].value = models[1]->P_TAU[k*N*M + i];
		}
		for (int i = 0; i < Z*N*M; i++, j++)
		{
			prob.x[K+k][j].index = j + 1;
			prob.x[K+k][j].value = models[1]->P_MU[k*Z*N*M + i];
		}
		for (int i = 0; i < Z*Z*N*M; i++, j++)
		{
			prob.x[K+k][j].index = j + 1;
			prob.x[K+k][j].value = models[1]->P_SIG[k*Z*Z*N*M + i];
		}
	}

	// set parameters
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 500;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	// Launch SVM here
	std::cout << "Before train" << std::endl;
	svm_model * model = svm_train(&prob, &param);
	std::cout << "After train" << std::endl;
	// End SVM here
	
	// after train

	// fill results

	// delete temporary derivatives
	delete d_PI_0;
	delete d_A_0;
	delete d_TAU_0;
	delete d_MU_0;
	delete d_SIG_0;
	delete d_PI_1;
	delete d_A_1;
	delete d_TAU_1;
	delete d_MU_1;
	delete d_SIG_1;
}

void HMM::calcDerivatives(real_t * observations, int nOfSequences, real_t * d_PI, real_t * d_A, real_t * d_TAU, real_t * d_MU, real_t * d_SIG)
{
	real_t * old_Otr = Otr;
	int old_K = K;
	Otr = observations;
	K = nOfSequences;

	// carry out some internal calculations 
	internal_calculations(-1);

	// calc derivative for each parameter and each sequence
	for (int k = 0; k<K; k++)
		calc_derivative(k, d_PI, d_A, d_TAU, d_MU, d_SIG);

	Otr = old_Otr;
	K = old_K;
}