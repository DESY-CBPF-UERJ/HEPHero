#include "HEPHero.h"

int main( int argc, char **argv ) {

    HEPHero *analysis = HEPHero::GetInstance( argv[1] );
    //cout << "Now initing:" << endl;
    analysis->Init();
    //cout << "Starting event loop:" << endl;
    analysis->RunEventLoop( (argc > 2 ) ? atoi(argv[2]) : -1 );
    //cout << "Finishing: " << endl;
    analysis->FinishRun();
    return 0;
}
