#pragma once
#include "HMM.h"
#include "fstream"

HMM::HMM(std::string filename)
{
	std::fstream f;
	f.open(filename+"Params.txt",std::fstream::in);
	f>>N>>M>>Z>>T>>K>>NumInit;
	f.close();

	//�������� ������ ��� ���������� ���
	PI = new real_t[N];				// ��������� ������������� ������������
	A = new real_t[N*N];				// ����������� ���������
	TAU = new real_t[N*M];			
	MU = new real_t[N*M*Z];
	SIG = new real_t[N*M*Z*Z];
	alf = new real_t[T*N*K];
	bet = new real_t[T*N*K];
	c = new real_t[T*K];				// ������������ ��������
	ksi = new real_t[(T-1)*N*N*K];	
	gam = new real_t[T*N*K];
	gamd = new real_t[T*N*M*K];
	alf_t = new real_t[T*N*K];
	bet_t = new real_t[T*N*K];
	B = new real_t[N*T*K];			// ����������� ��������� ����������

	//��������� �����������
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

	// �������������� ����� OpenCL
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

// ��������������� �������
void copyArray(real_t * dest, real_t * source, int n)
{
	for(int i=0; i<n; i++)
		dest[i]=source[i];
}

void HMM::findModelParameters()
{
	using namespace std;
	// �������� ����� ����� ��� ���� ��������� ����������� � �������� ������ ����� ����������
	real_t p, p0 = -1000000000000000.;
	real_t * PIw = new real_t[N];
	real_t * Aw = new real_t[N*N]; 
	real_t * TAUw = new real_t[N*M];
	real_t * MUw = new real_t[Z*N*M];
	real_t * SIGw = new real_t[Z*Z*N*M];
	// n - ����� �����������
	for(int n=0; n<NumInit; n++)
	{
		p = calcBaumWel�h(n);
		
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

void HMM::classifyObservations(real_t * p)
{
	using namespace std;
	// ���������� ����������
	internal_calculations(-1);

	// ������ 5
	//cout << "classification probabilities:" << endl;
	for(int k=0;k<K;k++){
		for(int t=0;t<T;t++)
			p[k]-=log(c(t,k));
		//cout << p[k] << endl;
	}
}

real_t HMM::g(int t,int k,int i,int m,int n)
{
	//�������� � ������������� ��������������� ���������
	real_t det=1.,res=0.;
	real_t tmp1,tmp2;
	if (n==-1) //������ � ��� ����������� ����������� ������ 
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

real_t HMM::calcBaumWel�h(int n)
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
		// ������� ���� ��������������� ����������
		internal_calculations(n);

		// ������ 3.1
		for(int i=0;i<N;i++)
		{
			gam_sum[i]=0.;
			for(int m=0;m<M;m++)
				gamd_sum[i*M+m]=0.;
		}
		
		// ������ 3.2
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

		
		// ������ 3.3
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
		
		// ������ 3.4
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

		// ������ 3.5
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				TAU(i,m)=gamd_sum[i*M+m]/gam_sum[i];

		// DEBUG - TAU - good
		/*f.open("debugging_TAU.txt", std::fstream::out);
		for (int i = 0; i<N*M; i++)
			f << TAU[i] << std::endl;
		f.close();*/
		// DEBUG
		
		// ������ 3.6
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

		// ������ 3.7
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
		
		// ������ 3.8
		for(int i=0;i<N;i++)
			PI1[i]=PI[i];
		// ������ 3.9
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++)
				A1(i,j)=A(i,j);
		// ������ 3.10
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				TAU1(i,m)=TAU(i,m);
		// ������ 3.11
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z=0;z<Z;z++)
					MU1(z,i,m,n)=MU(z,i,m);
		// ������ 3.12
		for(int i=0;i<N;i++)
			for(int m=0;m<M;m++)
				for(int z1=0;z1<Z;z1++)
					for(int z2=0;z2<Z;z2++)
						SIG1(z1,z2,i,m,n)=SIG(z1,z2,i,m);		

		p = calcProbability();		// ��������� ����� �����������
		std::swap(p, p_pred);		// �������� ����� � ������ ����������� �������
		std::cout << "diff =" << abs(p - p_pred) << std::endl;
	}
	
	std::cout << "Baum Welsh: iters = " << iter << std::endl;

	delete gamd_sum; delete gam_sum; delete tmp3;

	return p_pred;					// ������ ��������� �����������
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

	// ������ 1.1 (getB)
	for (int k=0;k<K;k++)
		for(int i=0;i<N;i++)
			for(int t=0;t<T;t++)
				B(i,t,k)=0;

	// ������ 1.2 (getB)
	for (int k=0;k<K;k++)
		for(int i=0;i<N;i++)
			for(int t=0;t<T;t++)
				for(int m=0;m<M;m++)
					B(i,t,k)+=TAU_used(i,m)*g(t,k,i,m,n);
	// ������!
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

	// ������ 2.1 (set_var)
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

	// ������ 2.2 (set_var)
	for(int k=0;k<K;k++)
	{
		for(int i=0;i<N;i++)
		{
			alf_t(0,i,k)=PI_used[i]*B(i,0,k);
			bet(T1,i,k)=1.;
		}
	}

	// ������ 2.3 (set_var)
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

	// ������ 2.4 (set_var)
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
	
	// DEBUG ��������
	/*if(abs(P-1.)>=0.1)
	{
		std::cout<<"error!!!!! p<>1!!!!\n";
		std::cout << std::endl;
	}*/

	///
	/// ����� ���������� ������ ��� ����� ��������
	///
	// ������ 2.5 (set_var)
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

		//��������! (set_var)
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



		// ������ 2.6 (set_var)
		for (int k = 0; k < K; k++)
		{
			for (int t = 0; t < T; t++)
			{
				for (int i = 0; i < N; i++)
				{
					gam(t, i, k) = alf(t, i, k)*bet(t, i, k);
					/*if(!_finite(gam(t,i,k)))	// ��������
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

		// ��������� ?? 
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

		// ������ 2.7 (set_var)
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

		//  �������� ?? 
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
	// ������4: ��� ������� =( ��� 2� ��������
	real_t res=0;
	for(int k=0;k<K;k++)
		for(int t=0;t<T;t++)
			res -= log(c(t,k));
	return res;
}

void HMM::getTestObserv(std::string fname)
{
	std::fstream f;
	f.open(fname,std::fstream::in);
	for(int k=0;k<K;++k)
		for(int t=0;t<T;t++)
			for (int z=0;z<Z;z++)
				f>>Otr(k,t,z);
	f.close();
}

