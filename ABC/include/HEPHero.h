#ifndef HEPHERO_H
#define HEPHERO_H

#include "HEPBase.h"
#include "CMS.h"
#include "ML.h"



using namespace std;

class HEPHero : public HEPBase {
    public:
        static HEPHero* GetInstance( char* configFileName );
        ~HEPHero() {}
        
        bool Init();
        void RunEventLoop( int nEventsMax = -1);
        void FinishRun();
    
    private:
        static HEPHero* _instance;

        bool RunRoutines();
        void PreRoutines();
        
        HEPHero() {}
        HEPHero( char* configFileName );

        void FillControlVariables( string key, string value);
        void VerticalSysSizes();
        void VerticalSys();
        void Weight_corrections();
        bool MC_processing();
        void SetupAna();
        bool AnaRegion();
        void AnaSelection();
        void AnaSystematic();
        void FinishAna();


        void SetupTest();
        bool TestRegion();
        void TestSelection();
        void TestSystematic();
        void FinishTest();
        // INSERT YOUR SELECTION HERE



    //=============================================================================================
    // ANALYSIS SETUP
    //=============================================================================================



    
    //=============================================================================================
    // INPUT TREE SETUP - NANOAOD
    //=============================================================================================
    private:

        //UInt_t    run;
        //UInt_t    luminosityBlock;
        //ULong64_t event;

        Float_t genWeight;

        
};

#endif
