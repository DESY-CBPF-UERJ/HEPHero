
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;


class JES {
    private:
        vector<float> eta_ranges = {-5.4, -5.0, -4.4, -4.0, -3.5, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.4, 5.0, 5.4};
        vector<float> pt_ranges = {9.0, 11.0, 13.5, 16.5, 19.5, 22.5, 26.0, 30.0, 34.5, 40.0, 46.0, 52.5, 60.0, 69.0, 79.0, 90.5, 105.5, 123.5, 143.0, 163.5, 185.0, 208.0, 232.5, 258.5, 286.0, 331.0, 396.0, 468.5, 549.5, 639.0, 738.0, 847.5, 968.5, 1102.0, 1249.5, 1412.0, 1590.5, 1787.0, 2003.0, 2241.0, 2503.0, 2790.5, 3107.0, 3455.0, 3837.0, 4257.0, 4719.0, 5226.5, 5784.0, 6538.0, 1000000};
        vector<vector<float>> values;
        
        int getIndex( vector<float> vec, float inputVar ){ // Return index in the vector
            int idx = 0;
            if( inputVar < vec[ vec.size()-1 ] ){
                for (unsigned int i = 0; i < vec.size()-1 ; i++){
                    if ( inputVar >= vec[i] && inputVar < vec[i+1]  ){
                        idx = i;
                        break;
                    }
                }
            }
            else idx = vec.size()-2;
            return idx;
        }
    public:
        void ReadFile( string JES_file ){
            ifstream in;
            in.open(JES_file);
            
            vector<float> values_line(pt_ranges.size()-1, 0);
            for(int i = 0; i < eta_ranges.size()-1; i++){
                values.push_back(values_line);
            }
            
            if (in.is_open()){
                for(int i = 0; i < eta_ranges.size()-1; i++){
                    for(int j = 0; j < pt_ranges.size()-1; j++){
                        in >> values[i][j];    
                    }
                }
                in.close();
            }else{
                cout << "Error opening " << JES_file << endl;
            }
        }
        
        float getUnc( float eta_input, float pt_input ){
            int eta_idx = getIndex( eta_ranges, eta_input ); 
            int pt_idx = getIndex( pt_ranges, pt_input ); 
            float uncertainty = values[eta_idx][pt_idx];
            return uncertainty;
        }
};


int main(){
    
    string JES_file = "JES_UNC_16.txt";
    
    JES JES_unc;
    JES_unc.ReadFile( JES_file );
    float test = JES_unc.getUnc( -4.8, 1200 );
    cout << "test: " << test1 << endl;
  
  return 0;
}







