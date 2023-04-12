#include <iostream>

#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TH1.h>
#include <TFile.h>
#include <vector>



class KinematicReconstruction_LSroutines{
    // Access specifier
    public:
         
        struct TopSolution{
            double dR;
            double dN;
            TLorentzVector top;
            TLorentzVector topbar;
            TLorentzVector neutrino;
            TLorentzVector neutrinobar;
            TLorentzVector wp;
            TLorentzVector wm;
            double x1;
            double x2;
            double mtt;
            double weight;
        };
         
        KinematicReconstruction_LSroutines() 
        {
            mt_    = TopMASS;
            mtbar_ = TopMASS;
            mb_    = 4.8;
            mbbar_ = 4.8;
            mw_    = 80.4;
            mwbar_ = 80.4; 
            ml_    = 0.;
            mal_    = 0.; 
            mv_ = 0.;
            mav_= 0.;
        }
         
        KinematicReconstruction_LSroutines(const double& mass_Wp, const double& mass_Wm)
        {
            mt_    = TopMASS;
            mtbar_ = TopMASS;
            mb_    = 4.8;
            mbbar_ = 4.8;
            mw_    = mass_Wp;
            mwbar_ = mass_Wm; 
            ml_    = 0.;
            mal_    = 0.;
            mv_= 0.;
            mav_= 0.;
        }
        KinematicReconstruction_LSroutines( const double& mass_top, const double& mass_topbar, 
                                            const double& mass_b, const double& mass_bbar, 
                                            const double& mass_Wp, const double& mass_Wm, 
                                            const double& mass_al, const double& mass_l)
        {
            mt_    = mass_top;
            mtbar_ = mass_topbar;
            mb_    = mass_b;
            mbbar_ = mass_bbar;
            mw_    = mass_Wp;
            mwbar_ = mass_Wm;
            mal_= mass_al;
            ml_= mass_l;
            mv_= 0.;
            mav_= 0.;
        }
         
        void setConstraints(TLorentzVector LV_al, 
                            TLorentzVector LV_l,
                            TLorentzVector LV_b,
                            TLorentzVector LV_bbar,
                            double missPx,
                            double missPy)
        {
            l_  = LV_l;
            al_ = LV_al;
            b_  = LV_b;
            bbar_ = LV_bbar;
            px_miss_ = missPx;
            py_miss_ = missPy;
            doAll();
        }
         
        int getNsol(){ 
            // Get the number of solution
            return nSol_;
        }
         
        const std::vector<TopSolution>* getTtSol()const 
        {
            return &ttSol_;
        }
         
    
    // Access specifier
    private:
        // Private attribute
        double TopMASS = 172.5;

        TLorentzVector met_;
        TLorentzVector al_;
        TLorentzVector l_;
        TLorentzVector b_;
        TLorentzVector bbar_;
        TLorentzVector top_;
        TLorentzVector topbar_;
        TLorentzVector neutrino_;
        TLorentzVector neutrinobar_;
        TLorentzVector w_;
        TLorentzVector wbar_;
        TLorentzVector tt_;
        
        TLorentzVector true_top_;
        TLorentzVector true_topbar_;
        TLorentzVector true_neutrino_;
        TLorentzVector true_neutrinobar_;
        
        double px_miss_;
        double py_miss_;

        
        double mt_; // t mass
        double mtbar_;  // tbar mass
        double mb_; // b mass
        double mbbar_; // bbar mass
        double mw_; // w mass
        double mwbar_; // wbar mass
        double ml_; // lepton mass 
        double mal_; // anti-lepton mass 
        double mv_; // neutrino mass 
        double mav_; // anti-neutrino mass

        // Lars Sonnenschein coefficients
        double a1_,a2_,a3_,a4_;
        double b1_,b2_,b3_,b4_;
        double c22_,c21_,c20_,c11_,c10_,c00_;
        double d22_,d21_,d20_,d11_,d10_,d00_;
        double d0_,d1_,d2_;
        double c0_,c1_,c2_;

        double coeffs_[5];
        std::vector<double> vect_pxv_;
        int nSol_;
        std::vector<TopSolution> ttSol_;

        // Member Functions()
         
        double sqr(const double& x){ 
            return (x*x);
        }
         
        int sign(const long double& ld)const  
        {
            if(fabs(ld)<0.0000000000001) return 0;
            return (ld>0)?1:-1;
        }
        
         
        void findCoeff( double* const koeficienty )  
        {
            a1_ = ((b_.E()+al_.E())*(mw_*mw_-mal_*mal_-mv_*mv_)-al_.E()*(mt_*mt_-mb_*mb_-mal_*mal_-mv_*mv_)+2*b_.E()*al_.E()*al_.E()-2*al_.E()*(al_.Vect().Dot(b_.Vect())))/(2*al_.E()*(b_.E()+al_.E()));
            a2_ = 2*(b_.E()*al_.Px()-al_.E()*b_.Px())/(2*al_.E()*(b_.E()+al_.E()));
            a3_ = 2*(b_.E()*al_.Py()-al_.E()*b_.Py())/(2*al_.E()*(b_.E()+al_.E()));
            a4_ = 2*(b_.E()*al_.Pz()-al_.E()*b_.Pz())/(2*al_.E()*(b_.E()+al_.E()));
                 
            b1_ = ((bbar_.E()+l_.E())*(mwbar_*mwbar_-ml_*ml_-mav_*mav_)-l_.E()*(mtbar_*mtbar_-mbbar_*mbbar_-ml_*ml_-mav_*mav_)+2*bbar_.E()*l_.E()*l_.E()-2*l_.E()*(l_.Vect().Dot(bbar_.Vect())))/(2*l_.E()*(bbar_.E()+l_.E()));
            b2_ = 2*(bbar_.E()*l_.Px()-l_.E()*bbar_.Px())/(2*l_.E()*(bbar_.E()+l_.E()));
            b3_ = 2*(bbar_.E()*l_.Py()-l_.E()*bbar_.Py())/(2*l_.E()*(bbar_.E()+l_.E()));
            b4_ = 2*(bbar_.E()*l_.Pz()-l_.E()*bbar_.Pz())/(2*l_.E()*(bbar_.E()+l_.E()));
                
            c22_ = (sqr((mw_*mw_-mal_*mal_-mv_*mv_))-4*(sqr(al_.E())-sqr(al_.Pz()))*sqr(a1_/a4_)-4*(mw_*mw_-mal_*mal_-mv_*mv_)*al_.Pz()*(a1_/a4_))/sqr(2*(b_.E()+al_.E())); 
            c21_ = (4*(mw_*mw_-mal_*mal_-mv_*mv_)*(al_.Px()-al_.Pz()*(a2_/a4_))-8*(sqr(al_.E())-sqr(al_.Pz()))*(a1_*a2_/sqr(a4_))-8*al_.Px()*al_.Pz()*(a1_/a4_))/sqr(2*(b_.E()+al_.E())); 
            c20_ = (-4*(sqr(al_.E())-sqr(al_.Px()))-4*(sqr(al_.E())-sqr(al_.Pz()))*sqr(a2_/a4_)-8*al_.Px()*al_.Pz()*(a2_/a4_))/sqr(2*(b_.E()+al_.E())); 
            c11_ = (4*(mw_*mw_-mal_*mal_-mv_*mv_)*(al_.Py()-al_.Pz()*(a3_/a4_))-8*(sqr(al_.E())-sqr(al_.Pz()))*(a1_*a3_/sqr(a4_))-8*al_.Py()*al_.Pz()*(a1_/a4_))/sqr(2*(b_.E()+al_.E())); 
            c10_ = (-8*(sqr(al_.E())-sqr(al_.Pz()))*(a2_*a3_/sqr(a4_)) + 8*al_.Px()*al_.Py() - 8*al_.Px()*al_.Pz()*(a3_/a4_) - 8*al_.Py()*al_.Pz()*(a2_/a4_))/sqr(2*(b_.E()+al_.E()));
            c00_ = (-4*(sqr(al_.E())-sqr(al_.Py())) -4*(sqr(al_.E())-sqr(al_.Pz()))*sqr(a3_/a4_)-8*al_.Py()*al_.Pz()*(a3_/a4_))/sqr(2*(b_.E()+al_.E()));
            
            double D22,D21,D20,D11,D10,D00;
            D22 = (sqr((mwbar_*mwbar_-ml_*ml_-mav_*mav_))-4*(sqr(l_.E())-sqr(l_.Pz()))*sqr(b1_/b4_)-4*(mwbar_*mwbar_-ml_*ml_-mav_*mav_)*l_.Pz()*(b1_/b4_))/sqr(2*(bbar_.E()+l_.E())); 
            D21 = (4*(mwbar_*mwbar_-ml_*ml_-mav_*mav_)*(l_.Px()-l_.Pz()*(b2_/b4_))-8*(sqr(l_.E())-sqr(l_.Pz()))*(b1_*b2_/sqr(b4_))-8*l_.Px()*l_.Pz()*(b1_/b4_))/sqr(2*(bbar_.E()+l_.E())); 
            D20 = (-4*(sqr(l_.E())-sqr(l_.Px()))-4*(sqr(l_.E())-sqr(l_.Pz()))*sqr(b2_/b4_)-8*l_.Px()*l_.Pz()*(b2_/b4_))/sqr(2*(bbar_.E()+l_.E())); 
            D11 = (4*(mwbar_*mwbar_-ml_*ml_-mav_*mav_)*(l_.Py()-l_.Pz()*(b3_/b4_))-8*(sqr(l_.E())-sqr(l_.Pz()))*(b1_*b3_/sqr(b4_))-8*l_.Py()*l_.Pz()*(b1_/b4_))/sqr(2*(bbar_.E()+l_.E())); 
            D10 = (-8*(sqr(l_.E())-sqr(l_.Pz()))*(b2_*b3_/sqr(b4_)) + 8*l_.Px()*l_.Py() - 8*l_.Px()*l_.Pz()*(b3_/b4_) - 8*l_.Py()*l_.Pz()*(b2_/b4_))/sqr(2*(bbar_.E()+l_.E()));
            D00  = (-4*(sqr(l_.E())-sqr(l_.Py())) -4*(sqr(l_.E())-sqr(l_.Pz()))*sqr(b3_/b4_)-8*l_.Py()*l_.Pz()*(b3_/b4_))/sqr(2*(bbar_.E()+l_.E()));
            
            d22_ = D22+sqr(px_miss_)*D20+sqr(py_miss_)*D00+px_miss_*py_miss_*D10+px_miss_*D21+py_miss_*D11;
            d21_ = -D21-2*px_miss_*D20-py_miss_*D10;
            d20_ = D20;
            d11_ = -D11-2*py_miss_*D00-px_miss_*D10;
            d10_ = D10;
            d00_  = D00;
            
            koeficienty[4] = sqr(c00_)*sqr(d22_)+c11_*d22_*(c11_*d00_-c00_*d11_)+c00_*c22_*(sqr(d11_)-2*d00_*d22_)+c22_*d00_*(c22_*d00_-c11_*d11_);
            koeficienty[3] = c00_*d21_*(2*c00_*d22_-c11_*d11_)+c00_*d11_*(2*c22_*d10_+c21_*d11_)+c22_*d00_*(2*c21_*d00_-c11_*d10_)-c00_*d22_*(c11_*d10_+c10_*d11_)-2*c00_*d00_*(c22_*d21_+c21_*d22_)-d00_*d11_*(c11_*c21_+c10_*c22_)+c11_*d00_*(c11_*d21_+2*c10_*d22_);
            koeficienty[2] = sqr(c00_)*(2*d22_*d20_+sqr(d21_))-c00_*d21_*(c11_*d10_+c10_*d11_)+c11_*d20_*(c11_*d00_-c00_*d11_)+c00_*d10_*(c22_*d10_-c10_*d22_)+c00_*d11_*(2*c21_*d10_+c20_*d11_)+(2*c22_*c20_+sqr(c21_))*sqr(d00_)-2*c00_*d00_*(c22_*d20_+c21_*d21_+c20_*d22_)+c10_*d00_*(2*c11_*d21_+c10_*d22_)-d00_*d10_*(c11_*c21_+c10_*c22_)-d00_*d11_*(c11_*c20_+c10_*c21_);
            // koeficienty[1] = c00_*d21_*(2*c00_*d20_-c10_*d10_)-c00_*d20_*(c11_*d10_+c10_*d11_)+c00_*d10_*(c21_*d10_+2*c20_*d11_)-2*c00_*d00_*(c21_*d20_+c20_*d21_)+c10_*d00_*(2*c11_*d20_+c10_*d21_)-c20_*d00_*(2*c21_*d00_-c10_*d11_)-d00_*d10_*(c11_*c20_+c10_*c21_);
            koeficienty[1] = c00_*d21_*(2*c00_*d20_-c10_*d10_)-c00_*d20_*(c11_*d10_+c10_*d11_)+c00_*d10_*(c21_*d10_+2*c20_*d11_)-2*c00_*d00_*(c21_*d20_+c20_*d21_)+c10_*d00_*(2*c11_*d20_+c10_*d21_)+c20_*d00_*(2*c21_*d00_-c10_*d11_)-d00_*d10_*(c11_*c20_+c10_*c21_);
            koeficienty[0] = sqr(c00_)*sqr(d20_)+c10_*d20_*(c10_*d00_-c00_*d10_)+c20_*d10_*(c00_*d10_-c10_*d00_)+c20_*d00_*(c20_*d00_-2*c00_*d20_);
            
        }
         
        void quartic_equation(  const double& h0, 
                                const double& h1, 
                                const double& h2, 
                                const double& h3, 
                                const double& h4, 
                                std::vector<double>& v)
        {

            std::vector<double> result;
            //printf("Koefs_in_f: %f %f %f %f %f\n",h0,h1,h2,h3,h4); //printout
            //printf("Koefs_norm_in_f: %f %f %f %f %f\n",h0/h0,h1/h0,h2/h0,h3/h0,h4/h0); //printout
            if(sign(a4_)==0||sign(b4_)==0){
                result.push_back(0);
                v.swap(result);    
            }
            else{
                //printf("else1\n"); //printout
                if(sign(h0)==0) {
                    cubic_equation(h1,h2,h3,h4,result);
                    v.swap(result);
                }
                else{
                    //printf("else2\n"); //printout
                    if(sign(h4)==0) {
                        cubic_equation(h0,h1,h2,h3,result);
                        result[0]=result[0]+1;
                        result.push_back(0);
                        v.swap(result);
                    }
                    else {
                    //printf("else3\n"); //printout
                        
                        double H1=h1/h0;
                        double H2=h2/h0;
                        double H3=h3/h0;
                        double H4=h4/h0;
                        double K1 = H2 -3*sqr(H1)/8;
                        double K2 = H3 + H1*sqr(H1)/8-H1*H2/2;
                        double K3 = H4-3*sqr(sqr(H1))/256+sqr(H1)*H2/16-H1*H3/4;
                        //printf("Koefs Ki: %f %f %10.10f\n",K1,K2,K3);//printout
                        if(sign(K3)==0){
                            cubic_equation(1,0,K1,K2,result);
                            for(int i=1;i<=result[0];++i){
                                result[i]=result[i]-H1/4;
                            }
                            result[0]=result[0]+1;
                            result.push_back(-H1/4);
                            v.swap(result);
                        }
                        else{
                            //printf("else4\n"); //printout
                            std::vector<double> result_t12;
                            std::vector<double> result_t1;
                            result_t1.push_back(0);
                            std::vector<double> result_t2;
                            result_t2.push_back(0);
                            cubic_equation(1,2*K1,(K1*K1-4*K3),(-1)*K2*K2,result_t12); 
                            
                            //std::cout << "hehehe:  " << result_t12[0]  <<  std::endl; //printout
                            for(int i=1;i<=result_t12[0];++i){
                                //std::cout << "heh:  " << result_t12[i]  << std::endl; //printout
                                if(result_t12[i]>=0){
                                    result_t1[0]=result_t1[0]+2;
                                    result_t1.push_back(sqrt(result_t12[i]));
                                    result_t1.push_back((-1)*sqrt(result_t12[i]));
                                    result_t2[0]=result_t2[0]+2;
                                    result_t2.push_back((K1+result_t12[i]-K2/sqrt(result_t12[i]))/2);
                                    result_t2.push_back((K1+result_t12[i]+K2/sqrt(result_t12[i]))/2);                                
                                }
                            }  
                            
                            std::vector<double> pre_result1;

                            result.push_back(0);
                            for(int i=1;i<=result_t1[0];++i){
                                //std::cout << "quadric_equation:   " << i << " " << result_t1[i] << " " << result_t2[i] << std::endl; //printout
                                
                                quadratic_equation(1,result_t1[i],result_t2[i],pre_result1);
                                
                                for(int j=1;j<=pre_result1[0];++j){
                                    // if(pre_result1[0]==2)std::cout << "quadric_equation:   " << i << " " << pre_result1[1] << " " << pre_result1[2] << std::endl; //printout

                                    int flag=1;
                                    for(int r=1;r<=result.at(0);++r){
                                        if(fabs(result.at(r)-pre_result1[j])<0.02)flag=0;
                                        //printf("Result-result: %10.10f  \n",result[r]-pre_result1[j]);
                                    }
                                    if(flag){
                                        result.at(0)=result.at(0)+1;
                                        result.push_back(pre_result1[j]);
                                    }
                                }
                                pre_result1.clear();                          
                            }                          
                            for(int k=1;k<=result.at(0);++k){
                                //printf("Result: %f   %f \n",H1/4,h1/4); //printout
                                result.at(k)=result.at(k)-H1/4;
                            }
                            v.swap(result);   
                        }
                    }
                }
            }
        }
         
        void cubic_equation(const double& a, 
                            const double& b, 
                            const double& c, 
                            const double& d, 
                            std::vector<double>& v)const
        {
            // std::cout <<  "--------------(debug)----------- cubic_equation Initiated "  << std::endl;
            std::vector<double> result;
            if(a==0){
                quadratic_equation(b,c,d,result);
                v.swap(result);
            }
            else {
                double s1 = b/a;
                double s2 = c/a;
                double s3 = d/a;
                
                double q = (s1*s1-3*s2)/9;
                double q3 = q*q*q;
                double r = (2*s1*s1*s1-9*s1*s2+27*s3)/54;
                double r2 = r*r;
                double S = r2-q3;

                if(sign(S)<0) {
                    result.push_back(3);
                    double F = acos(r/sqrt(q3));
                    result.push_back(-2*sqrt(fabs(q))*cos(F/3)-s1/3);
                    result.push_back(-2*sqrt(fabs(q))*cos((F+2*TMath::Pi())/3)-s1/3);
                    result.push_back(-2*sqrt(fabs(q))*cos((F-2*TMath::Pi())/3)-s1/3);  
                    v.swap(result);
                }
                else {
                    if(sign(S)==0) {
                            long double A = r+sqrt(fabs(r2-q3));
                            A = A<0 ? pow(fabs(A),(long double)1.0/3) : -pow(fabs(A),(long double)1.0/3);
                            long double B = sign(A) == 0 ? 0 : q/A; 
                            result.push_back(2);
                            result.push_back(A+B-s1/3);
                            result.push_back(-0.5*(A+B)-s1/3);  //!!!
                            v.swap(result);
                    }
                    else {
                        long double A = r+sqrt(fabs(r2-q3));
                        A = A<0 ? pow(fabs(A),(long double)1.0/3) : -pow(fabs(A),(long double)1.0/3);
                        long double B = sign(A) == 0 ? 0 : q/A; 
                        result.push_back(1);
                        result.push_back(A+B-s1/3);
                        v.swap(result);
                    }
                }
            }
        }

         
        void quadratic_equation(const double& a, 
                                const double& b, 
                                const double& c, 
                                std::vector<double>& v)const
        {
            // std::cout <<  "--------------(debug)----------- quadratic_equation Initiated "  << std::endl;

            std::vector<double> result;
            //printf("a: %10.10f\n",a);//printout
            if(a==0){
                linear_equation(b,c,result);
                v.swap(result);
            }
            else {
                double D = b*b-4*a*c;
                //printf("D: %10.10f\n",D);//printout
                if(sign(D)<0){
                    result.push_back(0);
                    v.swap(result);
                }
                else {
                    if(sign(D)==0){
                        result.push_back(1);
                        result.push_back((-1)*b/(2*a));
                        v.swap(result);
                    }
                    else{
                        result.push_back(2);
                        result.push_back((-b-sqrt(D))/(2*a));
                        result.push_back((-b+sqrt(D))/(2*a));
                        v.swap(result);
                    }
                }
            }
        }

         
        void linear_equation(   const double& a, 
                                const double& b, 
                                std::vector<double>& v)const
        {
            // std::cout <<  "--------------(debug)----------- linear_equation Initiated "  << std::endl;
            std::vector<double> result;
            if(a==0) {
                result.push_back(0);
                v.swap(result);
            }
            else {
                result.push_back(1);
                result.push_back((-1)*(b/a));
                v.swap(result);
            }
        }

         
        void topRec(const double& px_neutrino) 
        {
            // std::cout <<  "--------------(debug)----------- topRec Initiated "  << std::endl;
            double pxp, pyp, pzp, pup, pvp, pwp;
            
            d0_ = d00_;
            d1_ = d11_+d10_*px_neutrino;
            d2_ = d22_+d21_*px_neutrino+d20_*px_neutrino*px_neutrino;
            
            c0_ = c00_;
            c1_ = c11_+c10_*px_neutrino;
            c2_ = c22_+c21_*px_neutrino+c20_*px_neutrino*px_neutrino;
            
            pup = px_neutrino;
            pvp = (c0_*d2_-c2_*d0_)/(c1_*d0_-c0_*d1_);
            pwp = (-1)*(a1_+a2_*pup+a3_*pvp)/a4_;
            
            pxp = px_miss_-pup;   
            pyp = py_miss_-pvp;
            pzp = (-1)*(b1_+b2_*pxp+b3_*pyp)/b4_;
            
            neutrinobar_.SetXYZM(pxp, pyp, pzp, mav_);
            neutrino_.SetXYZM(pup, pvp, pwp, mv_);
                
            top_ = b_ + al_ + neutrino_;
            topbar_ = bbar_ + l_ + neutrinobar_; 
            tt_=top_+topbar_;
            w_ = al_ + neutrino_;
            wbar_ = l_ + neutrinobar_;
        }
         
        void doAll(){ 
            findCoeff(coeffs_);
            quartic_equation(coeffs_[0],coeffs_[1],coeffs_[2],coeffs_[3],coeffs_[4],vect_pxv_);
            nSol_ = vect_pxv_[0];

            for(int i=1;i<=nSol_;++i)
            {
                topRec(vect_pxv_[i]);
                TopSolution TS_temp;
                
                TS_temp.top = top_;
                TS_temp.topbar = topbar_;
                TS_temp.wp = w_;
                TS_temp.wm = wbar_;
                TS_temp.neutrino = neutrino_;
                TS_temp.neutrinobar = neutrinobar_;
                
                //proton Energy [GeV]  
                double protonE = 6500; //FIXME:  set as global variable and which value ????????
                TS_temp.x1 = (top_.E()+topbar_.E()+top_.Pz()+topbar_.Pz())/(2*protonE);
                TS_temp.x2 = (top_.E()+topbar_.E()-top_.Pz()-topbar_.Pz())/(2*protonE);
                TS_temp.mtt = tt_.M();
                
                TS_temp.weight = 1.0/tt_.M();
                // TS_temp.weight=Landau2D(neutrino_.E(),neutrinobar_.E());
                
                // std::cout << "tt_ " << tt_.M() << "  TS_temp.weight " << TS_temp.weight  << std::endl;
                ttSol_.push_back(TS_temp);
            }

            nSol_=ttSol_.size();
            // std::cout << "nSol_ " << nSol_ << std::endl;
            if(nSol_>0) sortTopSol(ttSol_);

        }
         
        void sortTopSol(std::vector<TopSolution>& v)const  
        {
            for(uint i=0;i<v.size()-1;++i) {
                if(v[i].weight < v[i+1].weight){
                    swapTopSol(v[i],v[i+1]);
                    i=-1;
                }
            }
            //v.swap(result);
        }
         
        void swapTopSol(TopSolution& sol1, 
                        TopSolution& sol2)const   
        {   
            TopSolution aux = sol1;
            sol1 = sol2;
            sol2 = aux;
        }
         
};

//=======================================================
//=======================================================

class KinematicReconstruction_MeanSol{
    // Access specifier
    private:
        // Private attribute
         
        std::vector<TLorentzVector> v_top_;
        std::vector<TLorentzVector> v_topbar_;
        std::vector<TLorentzVector> v_n_;
        std::vector<TLorentzVector> v_nbar_;
        
        std::vector<double> v_weight_;
        double sum_weight_;  
        double max_sum_weight_;
        
        double mass_top_;
        // const double mass_top_;
        

    // Access specifier
    public:
        // KinematicReconstruction_MeanSol(const double& topm):
        // sum_weight_(0),
        // max_sum_weight_(0),
        // mass_top_(topm)
        // {}
        //  
        // ~KinematicReconstruction_MeanSol()
        // {}
         
        KinematicReconstruction_MeanSol(const double& topm) 
        {
            sum_weight_ = 0;
            max_sum_weight_ = 0;
            mass_top_ = topm;
        }
         
        void clear()  
        {
            v_top_.clear();
            v_topbar_.clear();
            v_n_.clear();
            v_nbar_.clear();
            v_weight_.clear();
            sum_weight_ = 0;
            max_sum_weight_ = 0;
        }
         
        void add(   const TLorentzVector& top, 
                    const TLorentzVector& topbar, 
                    const TLorentzVector& n, 
                    const TLorentzVector& nbar,
                    const double& weight,
                    const double& mbl_weight)
        {
            v_top_.push_back(top);
            v_topbar_.push_back(topbar);
            v_n_.push_back(n);
            v_nbar_.push_back(nbar);
            v_weight_.push_back(weight);
            
            sum_weight_ = sum_weight_ + weight;
            max_sum_weight_ = max_sum_weight_ + mbl_weight;
        }
         
        void add(const TLorentzVector& top, 
                const TLorentzVector& topbar,
                const TLorentzVector& n,
                const TLorentzVector& nbar,
                const double& weight)
        {
            v_top_.push_back(top);
            v_topbar_.push_back(topbar);
            v_n_.push_back(n);
            v_nbar_.push_back(nbar);
            v_weight_.push_back(weight);

            sum_weight_ = sum_weight_ + weight;
            max_sum_weight_ = max_sum_weight_ +weight;
        }
         
        void getMeanVect(   TLorentzVector& lv, 
                            const std::vector<TLorentzVector>& vlv, 
                            const double& mass) const
        {
            double px_sum=0;
            double py_sum=0;
            double pz_sum=0;
            double px=0;
            double py=0;
            double pz=0;
            
            for(int i=0;i<((int)vlv.size());++i)
            {
                px_sum=px_sum+v_weight_.at(i)*vlv.at(i).Px();
                py_sum=py_sum+v_weight_.at(i)*vlv.at(i).Py();
                pz_sum=pz_sum+v_weight_.at(i)*vlv.at(i).Pz();
            }
            
            px=px_sum/sum_weight_;
            py=py_sum/sum_weight_;
            pz=pz_sum/sum_weight_;
            
            lv.SetXYZM(px,py,pz,mass);
        }
         
        void getMeanSol(TLorentzVector& top, 
                        TLorentzVector& topbar, 
                        TLorentzVector& n, 
                        TLorentzVector& nbar)const
        {
            getMeanVect(top,v_top_,mass_top_);
            getMeanVect(topbar,v_topbar_,mass_top_);
            getMeanVect(n,v_n_,0);
            getMeanVect(nbar,v_nbar_,0);
            
        }
         
        double getSumWeight()const 
        {
            return sum_weight_; // for 1 weight
        //     return max_sum_weight_; // for 2 weights
            
        }
         
        int getNsol()const 
        {
            return (int)v_top_.size();
        }
   
};


//=======================================================
//=======================================================
class KinematicReconstruction{
    // Access specifier



    // Access specifier
    private:

        int nSol_;
        double TopMASS = 172.5;
        TRandom3* r3_;
    
        struct Struct_KinematicReconstruction{
            TLorentzVector lp, lm;
            TLorentzVector jetB, jetBbar;
            size_t jetB_index, jetBbar_index;
            TLorentzVector met;
            TLorentzVector Wplus, Wminus;
            TLorentzVector top, topBar;
            TLorentzVector neutrino, neutrinoBar;
            TLorentzVector ttbar;
            double recMtop;
            double weight;
            int ntags;
        };

        // W mass
        TH1* h_wmass_;
        // jet resolution
        TH1* h_jetAngleRes_;
        TH1* h_jetEres_;
        //lepton resolution
        TH1* h_lepAngleRes_;
        TH1* h_lepEres_;
        // mbl
        TH1* h_mbl_w_;

         
        Struct_KinematicReconstruction sol_;
        std::vector<Struct_KinematicReconstruction> sols_;

        bool NoSmearingFlag = true;

        void loadData()   // KinematicReconstruction
        {
            // Read all histograms for smearings from file
            //std::cout<<"Smearing requires input distributions from files\n";

            if (NoSmearingFlag) 
                r3_ = new TRandom3();
            else{
                r3_ = new TRandom3();

                TString data_path1 = "data";
                // https://gitlab.cern.ch/cms-desy-top/TopAnalysis/-/tree/Htottbar_2016/Configuration/analysis/common/data
                data_path1.Append("/KinReco_input_run2b_50ns.root");
                // if(era_ == Era::run1_8tev) data_path1.Append("/KinReco_input.root");
                // else if(era_ == Era::run2_13tev_50ns) data_path1.Append("/KinReco_input_run2b_50ns.root");
                // else if(era_ == Era::run2_13tev_25ns || era_ == Era::run2_13tev_2015_25ns) data_path1.Append("/KinReco_input_2015_Run2CD_25ns_76X.root");
                // else if(era_ == Era::run2_13tev_2016_25ns) data_path1.Append("/KinReco_input_2016_Run2BtoH_25ns_80X_v037_dilep_sel_25Sep2017_NLODY_FineEleID.root");
                // else{
                //     std::cerr<<"ERROR in KinematicReconstruction::loadData() ! Era is not supported: "
                //                  <<Era::convert(era_)<<"\n...break\n"<<std::endl;
                TFile dataFile(data_path1);

                //jet angle resolution
                h_jetAngleRes_ = (TH1F*)dataFile.Get("KinReco_d_angle_jet_step7");
                h_jetAngleRes_->SetDirectory(0);
                //jet energy resolution
                h_jetEres_ = (TH1F*)dataFile.Get("KinReco_fE_jet_step7");
                h_jetEres_->SetDirectory(0);
                //lep angle resolution
                h_lepAngleRes_ = (TH1F*)dataFile.Get("KinReco_d_angle_lep_step7");
                h_lepAngleRes_->SetDirectory(0);
                //lep energy resolution
                h_lepEres_ = (TH1F*)dataFile.Get("KinReco_fE_lep_step7");
                h_lepEres_->SetDirectory(0);
                //mbl mass
                h_mbl_w_ = (TH1F*)dataFile.Get("KinReco_mbl_true_step0");
                h_mbl_w_->SetDirectory(0);
                //h_mbl_w_ = (TH1F*)dataFile.Get("KinReco_mbl_true_wrong_step0");
                //h_mbl_w_->SetDirectory(0);
                // W mass
                h_wmass_ = (TH1F*)dataFile.Get("KinReco_W_mass_step0");
                h_wmass_->SetDirectory(0);
            }
        }
         
        void setRandomNumberSeeds(  TLorentzVector lepton,  // KinematicReconstruction
                                    TLorentzVector antiLepton , 
                                    TLorentzVector jet1, 
                                    TLorentzVector jet2)
        {
            // Asymmetric treatment of both jets and also both leptons, to ensure different seed for each combination
            const unsigned int seed = static_cast<int>( 1.e6*(jet1.Pt()/jet2.Pt()) * std::sin((lepton.Pt() + 2.*antiLepton.Pt())*1.e6) );
            gRandom->SetSeed(seed);
            r3_->SetSeed(seed);
        }
         
        void angle_rot( const double& alpha,  // KinematicReconstruction
                        const double& e, 
                        const TLorentzVector& inJet, 
                        TLorentzVector& jet_sm)const
        {
            double px_1, py_1, pz_1; // Coordinate system where momentum is along Z-axis
            
            //Transition matrix detector -> syst1 ...
            double x1, y1, z1;
            double x2, y2, z2;
            double x3, y3, z3;
            // ...
            
            TLorentzVector jet = inJet;
            if(fabs(jet.Px())<=e){jet.SetPx(0);}
            if(fabs(jet.Py())<=e){jet.SetPy(0);}
            if(fabs(jet.Pz())<=e){jet.SetPz(0);}
            
            //Rotation in syst 1 ...
            double phi = 2*TMath::Pi()*r3_->Rndm();
            pz_1 = jet.Vect().Mag()*cos(alpha);
            px_1 = - jet.Vect().Mag()*sin(alpha)*sin(phi);
            py_1 = jet.Vect().Mag()*sin(alpha)*cos(phi);  
            // ...
            
            //Transition detector <- syst1 ...
            if (jet.Py()!=0||jet.Pz()!=0)            {
                double d = sqrt(jet.Pz()*jet.Pz() + jet.Py()*jet.Py());
                double p = jet.Vect().Mag();
                
                x1 = d/p;
                y1 = 0;
                z1 = jet.Px()/p;
                
                x2 = - jet.Px()*jet.Py()/d/p;
                y2 = jet.Pz()/d;
                z2 = jet.Py()/p;
                
                x3 = - jet.Px()*jet.Pz()/d/p;
                y3 = - jet.Py()/d;
                z3 = jet.Pz()/p;
                
                jet_sm.SetPx(x1*px_1+y1*py_1+z1*pz_1);
                jet_sm.SetPy(x2*px_1+y2*py_1+z2*pz_1);
                jet_sm.SetPz(x3*px_1+y3*py_1+z3*pz_1);
                jet_sm.SetE(jet.E());
            }
            
            if (jet.Px()==0&&jet.Py()==0&&jet.Pz()==0) {
                jet_sm.SetPx(jet.Px());
                jet_sm.SetPy(jet.Py());
                jet_sm.SetPz(jet.Pz());
                jet_sm.SetE(jet.E());
            }
            
            if (jet.Px()!=0&&jet.Py()==0&&jet.Pz()==0) {
                jet_sm.SetPx(pz_1);
                jet_sm.SetPy(px_1);
                jet_sm.SetPz(py_1);
                jet_sm.SetE(jet.E());
            }
            // ...
        }

    public:
    
         
        bool solutionSmearing(  KinematicReconstruction_MeanSol& meanSolution,
                                TLorentzVector LV_al,  // KinematicReconstruction
                                TLorentzVector LV_l,
                                TLorentzVector LV_b,
                                TLorentzVector LV_bbar,
                                TLorentzVector met )
        {

            TLorentzVector al_temp = LV_al;
            TLorentzVector l_temp = LV_l ;
            TLorentzVector b_temp = LV_b;
            TLorentzVector bbar_temp = LV_bbar;
            TLorentzVector met_temp = met;

            // Read all histograms for smearings from file
            loadData();

            if (NoSmearingFlag){
                // If the mass of the lepton + bjet is bigger than 180 GeV It do not have solution
                if((al_temp+b_temp).M()>180. || (l_temp+bbar_temp).M()>180.) return false;

                bool isHaveSol(false);

                // Get the vector component of TLorentzVector
                TVector3 vX_reco =  - b_temp.Vect() - bbar_temp.Vect() - l_temp.Vect() - al_temp.Vect() - met_temp.Vect();

                TLorentzVector al_sm = al_temp;
                TLorentzVector l_sm = l_temp;
                TLorentzVector b_sm = b_temp;
                TLorentzVector bbar_sm = bbar_temp;
                TLorentzVector met_sm;
                // KinematicReconstruction_LSroutines( const double& mass_top, const double& mass_topbar, 
                //                                 const double& mass_b, const double& mass_bbar, 
                //                                 const double& mass_Wp, const double& mass_Wm, 
                //                                 const double& mass_al, const double& mass_l)

                KinematicReconstruction_LSroutines tp_sm( TopMASS, TopMASS, 4.8, 4.8, 80.4, 80.4, 0., 0.);
                // KinematicReconstruction_LSroutines tp_sm( TopMASS, TopMASS, b_sm.M(), bbar_sm.M(), 80.4, 80.4, al_sm.M(), l_sm.M());
                tp_sm.setConstraints(al_sm, l_sm, b_sm, bbar_sm, met_sm.Px(), met_sm.Py());

                if(tp_sm.getNsol()>0)
                {
                    isHaveSol = true;

                    // This information is from a histogram used in Smearing process (see below).
                    double bins[101] = {0.0, 1.8, 3.6, 5.4, 7.2, 9.0, 10.8, 12.6, 14.4, 16.2, 18.0, 19.8, 21.6, 23.4, 25.2, 27.0, 28.8, 30.6, 32.4, 34.2, 36.0, 37.8, 39.6, 41.4, 43.2, 45.0, 46.8, 48.6, 50.4, 52.2, 54.0, 55.8, 57.6, 59.4, 61.2, 63.0, 64.8, 66.6, 68.4, 70.2, 72.0, 73.8, 75.6, 77.4, 79.2, 81.0, 82.8, 84.6, 86.4, 88.2, 90.0, 91.8, 93.6, 95.4, 97.2, 99.0, 100.8, 102.6, 104.4, 106.2, 108.0, 109.8, 111.6, 113.4, 115.2, 117.0, 118.8, 120.6, 122.4, 124.2, 126.0, 127.8, 129.6, 131.4, 133.2, 135.0, 136.8, 138.6, 140.4, 142.2, 144.0, 145.8, 147.6, 149.4, 151.2, 153.0, 154.8, 156.6, 158.4, 160.2, 162.0, 163.8, 165.6, 167.4, 169.2, 171.0, 172.8, 174.6, 176.4, 178.2, 180.0};
                    double bin_content[100] = { 0.0, 51.76727, 1628.316, 2270.383, 2717.321, 3371.368, 3812.549, 4366.436, 5004.273, 5360.336, 6100.904, 6606.438, 7204.748, 7784.245, 8200.87, 9006.221, 9667.516, 10140.45, 10594.04, 11345.34, 12272.19, 12709.54, 13610.64, 14192.77, 15058.98, 15578.98, 16552.09, 17288.62, 18107.27, 18903.07, 19767.85, 20658.06, 21171.68, 22178.43, 22858.82, 23747.67, 24756.42, 25712.46, 26433.51, 27066.43, 27800.82, 28911.3, 29471.02, 30581.94, 31363.47, 31957.12, 32695.83, 33462.55, 34211.54, 34983.5, 35356.17, 36246.31, 36930.53, 36986.46, 37454.2, 38427.49, 38010.69, 38675.33, 38894.91, 38378.69, 38618.02, 38875.74, 38630.55, 38167.97, 37519.78, 37107.17, 36039.35, 35506.51, 34323.61, 33518.16, 32312.47, 31160.89, 29753.3, 27703.33, 25851.2, 23499.57, 21201.28, 18685.36, 16402.27, 13278.63, 10494.83, 7601.976, 4781.021, 2566.072, 1454.804, 858.0482, 686.4762, 461.5827, 406.4781, 303.8162, 266.0213, 227.0641, 164.1227, 168.7554, 153.7067, 127.0277, 112.9433, 93.59881, 75.74943, 564.683};

                    // FIXME: this loop is processed only once by definition, what is it needed for?
                    for(int i=0; i<=tp_sm.getNsol()*0; ++i){

                        double mass_temp = (al_sm+b_sm).M();
                        double mass_temp2 = (l_sm+bbar_sm).M();
                        
                        int idx1 = 0;
                        int idx2 = 0;
                        // loop array
                        for (int i = 0; i < 100; i++) {
                            if ( mass_temp > bins[i] ) idx1 = i;
                        }
                        for (int i = 0; i < 99; i++) {
                            if ( mass_temp2 > bins[i] ) idx2 = i;
                        }

                        double mbl_weight = bin_content[idx1] * bin_content[idx2] / 100000000;
                        // double mbl_weight = h_mbl_w_->GetBinContent(h_mbl_w_->FindBin((al_sm+b_sm).M())) * h_mbl_w_->GetBinContent(h_mbl_w_->FindBin((l_sm+bbar_sm).M())) /100000000;

                        meanSolution.add(   tp_sm.getTtSol()->at(i).top,
                                            tp_sm.getTtSol()->at(i).topbar,
                                            tp_sm.getTtSol()->at(i).neutrino,
                                            tp_sm.getTtSol()->at(i).neutrinobar,
                                            mbl_weight);
                    }
                }
                return isHaveSol;
            }
            else{
                // Set random number generator seeds
                setRandomNumberSeeds(l_temp, al_temp, b_temp, bbar_temp); 

                // If the mass of the lepton + bjet is bigger than 180 GeV It do not have solution
                if((al_temp+b_temp).M()>180. || (l_temp+bbar_temp).M()>180.) return false;

                bool isHaveSol(false);

                // Get the vector component of TLorentzVector
                TVector3 vX_reco =  - b_temp.Vect() - bbar_temp.Vect() - l_temp.Vect() - al_temp.Vect() - met_temp.Vect();

                // smearing for "sm" times to 
                for(int sm=0; sm<100; ++sm){

                    TLorentzVector b_sm = b_temp;
                    TLorentzVector bbar_sm = bbar_temp;
                    TLorentzVector met_sm;
                    TLorentzVector l_sm = l_temp;
                    TLorentzVector al_sm = al_temp;

                    //jets energy smearing
                    float fB = h_jetEres_->GetRandom();//fB=1;  //sm off
                    float xB = sqrt((fB*fB*b_sm.E()*b_sm.E()-b_sm.M2())/(b_sm.P()*b_sm.P()));
                    float fBbar = h_jetEres_->GetRandom();//fBbar=1; //sm off
                    float xBbar = sqrt((fBbar*fBbar*bbar_sm.E()*bbar_sm.E()-bbar_sm.M2())/(bbar_sm.P()*bbar_sm.P()));

                    //leptons energy smearing
                    float fL=h_lepEres_->GetRandom();//fL=1; //sm off
                    float xL=sqrt((fL*fL*l_sm.E()*l_sm.E()-l_sm.M2())/(l_sm.P()*l_sm.P()));
                    float faL=h_lepEres_->GetRandom();//faL=1;  //sm off
                    float xaL=sqrt((faL*faL*al_sm.E()*al_sm.E()-al_sm.M2())/(al_sm.P()*al_sm.P()));

                    //b-jet angle smearing
                    b_sm.SetXYZT(b_sm.Px()*xB,b_sm.Py()*xB,b_sm.Pz()*xB,b_sm.E()*fB);
                    angle_rot(h_jetAngleRes_->GetRandom(),0.001,b_sm,b_sm);
                    // angle_rot(const double& alpha, const double& e, const TLorentzVector& inJet, TLorentzVector& jet_sm)const

                    //bbar jet angel smearing
                    bbar_sm.SetXYZT(bbar_sm.Px()*xBbar,bbar_sm.Py()*xBbar,bbar_sm.Pz()*xBbar,bbar_sm.E()*fBbar);
                    angle_rot(h_jetAngleRes_->GetRandom(),0.001,bbar_sm,bbar_sm);

                    //lepton angle smearing
                    l_sm.SetXYZT(l_sm.Px()*xL,l_sm.Py()*xL,l_sm.Pz()*xL,l_sm.E()*fL);
                    angle_rot(h_lepAngleRes_->GetRandom(),0.001,l_sm,l_sm);

                    // anti lepton angle smearing
                    al_sm.SetXYZT(al_sm.Px()*xaL,al_sm.Py()*xaL,al_sm.Pz()*xaL,al_sm.E()*faL);
                    angle_rot(h_lepAngleRes_->GetRandom(),0.001,al_sm,al_sm);


                    TVector3 metV3_sm= -b_sm.Vect()-bbar_sm.Vect()-l_sm.Vect()-al_sm.Vect()-vX_reco;
                    met_sm.SetXYZM(metV3_sm.Px(),metV3_sm.Py(),0,0);
                    // std::cout << " l_sm.Pt() " << l_sm.Pt() << " l_sm.Eta() " << l_sm.Eta() << " l_sm.Phi() " << l_sm.Phi() << " l_sm.M() " << l_sm.M() <<   std::endl;
                    // std::cout << " met_sm.Pt() " << met_sm.Pt() << " met_sm.Eta() " << met_sm.Eta() << " met_sm.Phi() " << met_sm.Phi() << " met_sm.M() " << met_sm.M() <<   std::endl;

                    KinematicReconstruction_LSroutines tp_sm(h_wmass_->GetRandom(),h_wmass_->GetRandom());
                    tp_sm.setConstraints(al_sm, l_sm, b_sm, bbar_sm, met_sm.Px(), met_sm.Py());
                    // std::cout << " tp_sm.getNsol()  " << tp_sm.getNsol() << std::endl;

                    if(tp_sm.getNsol()>0)
                    {
                        isHaveSol = true;
                        // FIXME: this loop is processed only once by definition, what is it needed for?
                        for(int i=0; i<=tp_sm.getNsol()*0; ++i){
                            TLorentzVector testando1= tp_sm.getTtSol()->at(i).top;
                            TLorentzVector testando2 = tp_sm.getTtSol()->at(i).topbar;
                            double mbl_weight = h_mbl_w_->GetBinContent(h_mbl_w_->FindBin((al_sm+b_sm).M())) * h_mbl_w_->GetBinContent(h_mbl_w_->FindBin((l_sm+bbar_sm).M())) /100000000;
                            // std::cout << " mbl_weight  " << mbl_weight << "    ttbar sm.M() " << (testando1+testando2).M()  <<std::endl;
                            meanSolution.add(   tp_sm.getTtSol()->at(i).top,
                                                tp_sm.getTtSol()->at(i).topbar,
                                                tp_sm.getTtSol()->at(i).neutrino,
                                                tp_sm.getTtSol()->at(i).neutrinobar,
                                                mbl_weight);
                        }
                    } 
                }
                return isHaveSol;
            }
            
        }
        
         
        // void kinReco(const LV& leptonMinus, const LV& leptonPlus, const VLV* jets, const std::vector<double>* btags, const LV* met)
        void kinReco(   TLorentzVector leptonMinus, 
                        TLorentzVector leptonPlus, 
                        vector<TLorentzVector> jets, 
                        TLorentzVector met
                        )
        {
            sols_.clear();

            // loop over 1st jet
            for(std::vector<TLorentzVector>::const_iterator jet1 = jets.begin(); jet1 != jets.end(); jet1++)
            {
                KinematicReconstruction_MeanSol meanSolution(TopMASS);
                // std::cout << " --------------------------------------------------- " <<  std::endl;
                // loop over 2nd jet
                for(std::vector<TLorentzVector>::const_iterator jet2 = jets.begin(); jet2 != jets.end(); jet2++)
                {
                    // skip same jets
                    if(jet1 == jet2) continue;
                    
                    TLorentzVector jetB, jetBbar;
                    if(jet1->M() < 0)
                    {
                        TLorentzVector jet;
                        jet.SetPtEtaPhiM(jet1->Pt(), jet1->Eta(), jet1->Phi(), -1 * jet1->M());
                        jetB = jet;
                    }
                    else jetB = *jet1;

                    if(jet2->M() < 0)
                    {
                        TLorentzVector jet;
                        jet.SetPtEtaPhiM(jet2->Pt(), jet2->Eta(), jet2->Phi(), -1 * jet2->M());
                        jetBbar = jet;
                    }
                    else jetBbar = *jet2;

                    const bool hasSolution(solutionSmearing(meanSolution, leptonMinus, leptonPlus, jetB, jetBbar, met));
                    int numberOfBtags = 2;  // FIX: Automatizar

                    if(hasSolution){
                        meanSolution.getMeanSol(sol_.top, sol_.topBar, sol_.neutrino, sol_.neutrinoBar);
                        sol_.weight = meanSolution.getSumWeight();
                        sol_.Wplus = sol_.lp + sol_.neutrino;
                        sol_.Wminus = sol_.lm + sol_.neutrinoBar;
                        sol_.ttbar = sol_.top + sol_.topBar;
                        // sol_.jetB_index = bjetIndex;
                        // sol_.jetBbar_index = antiBjetIndex;
                        sol_.ntags = numberOfBtags;
                        sols_.push_back(sol_);
                        // std::cout << " meanSolution.getSumWeight()  " << meanSolution.getSumWeight() << std::endl;
                        
                    }
                    meanSolution.clear();

                }
            }
            setSolutions();

        }
        
        void setSolutions() {

            nSol_ = (int)(sols_.size());
            // std::cout << " nSol_  " << nSol_ << std::endl;
            
            if(nSol_ > 0){
                std::nth_element(begin(sols_), begin(sols_), end(sols_),
                                [](const Struct_KinematicReconstruction& a, const Struct_KinematicReconstruction& b){
                                    return  b.ntags < a.ntags || (b.ntags == a.ntags && b.weight < a.weight);
                                });
                
                sol_ = sols_[0];
            }
        }
         
        int getNSol()const
        {
            return nSol_;
        }
         
        Struct_KinematicReconstruction getSol()const
        {
            return sol_;
        }
         
        double getTtbarMass()const
        {
            return sol_.ttbar.M();
        }
         
        vector<TLorentzVector> getLorentzVector()const
        {
            vector<TLorentzVector> vec;
            vec.push_back(sol_.top);
            vec.push_back(sol_.topBar);
            return vec;
        }
         
        double getWeight()const
        {
            return sol_.weight;
        }
        //  
        // struct Struct_KinematicReconstruction{
        //     TLorentzVector lp, lm;
        //     TLorentzVector jetB, jetBbar;
        //     size_t jetB_index, jetBbar_index;
        //     TLorentzVector met;
        //     TLorentzVector Wplus, Wminus;
        //     TLorentzVector top, topBar;
        //     TLorentzVector neutrino, neutrinoBar;
        //     TLorentzVector ttbar;
        //     double recMtop;
        //     double weight;
        //     int ntags;
        // };

};
