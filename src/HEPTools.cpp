#include "HEPBase.h"


//---------------------------------------------------------------------------------------------
// PRINT INFO ABOUT THE SELECTION PROCESS
//---------------------------------------------------------------------------------------------
void HEPBase::WriteCutflowInfo(){

    if( SumGenWeights != SumGenWeights_original ){
        PROC_XSEC = (SumGenWeights/SumGenWeights_original)*PROC_XSEC;
    }

    if( _applyEventWeights || dataset_group == "Data"){
        luminosity = DATA_LUMI;
        lumi_total_unc = DATA_LUMI_TOTAL_UNC;
        for( unsigned int ilumi = 0; ilumi < DATA_LUMI_TAGS_UNC.size(); ++ilumi ) {
            lumi_values_unc.push_back(DATA_LUMI_VALUES_UNC.at(ilumi));
        }
    }else{
        luminosity = SumGenWeights/PROC_XSEC;
        lumi_total_unc = 0;
        for( unsigned int ilumi = 0; ilumi < DATA_LUMI_TAGS_UNC.size(); ++ilumi ) {
            lumi_values_unc.push_back(0);
        }
    }

    if( _sysID_lateral == 0 ) {
        _CutflowFile.open( _CutflowFileName.c_str(), ios::app );
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        _CutflowFile << "Luminosity: " << luminosity << " pb-¹" << endl;

        _CutflowFile.width(25); _CutflowFile << left << "Lumi. Unc. Tags:" << setw(10) << "total" << " = ";
        for( unsigned int ilumi = 0; ilumi < DATA_LUMI_TAGS_UNC.size(); ++ilumi ) {
            _CutflowFile << setw(10) << DATA_LUMI_TAGS_UNC.at(ilumi) << " ";
        }
        _CutflowFile << endl;

        _CutflowFile.width(25); _CutflowFile << left << "Lumi. Unc. Values [%]:" << setw(10) << lumi_total_unc << " = ";
        for( unsigned int ilumi = 0; ilumi < DATA_LUMI_TAGS_UNC.size(); ++ilumi ) {
            _CutflowFile << setw(10) << lumi_values_unc.at(ilumi) << " ";
        }
        _CutflowFile << endl;

        _CutflowFile << "Cross section: " << setprecision(16) << PROC_XSEC << " pb" << endl;
        _CutflowFile << "Sum of genWeights: " << setprecision(16) << SumGenWeights << endl;
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        _CutflowFile.width(20); _CutflowFile << left << " Cutflow" << " " << setw(20) << setprecision(16) << "Selected Events"
        << " " << setw(20) << setprecision(12) << "Stat. Error" << setw(15) << setprecision(6) << "Efficiency (%)" << endl;
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        int icut = 0;
        for( map<string,double>::iterator itr = _cutFlow.begin(); itr != _cutFlow.end(); ++itr ) {
            _CutflowFile.width(20); _CutflowFile << left << "|" + (*itr).first << " " << setw(20) << setprecision(16) << (*itr).second
            << " " << setw(20) << setprecision(12) << _StatisticalError.at(icut)
            << setw(15) << setprecision(6) << (*itr).second*100./SumGenWeights << endl;
            ++icut;
        }
        _CutflowFile.close();
    }
}



//---------------------------------------------------------------------------------------------
// Setup HDF maps
//---------------------------------------------------------------------------------------------
void HEPBase::HDF_insert( string varname, int* variable ) {
    _hdf_int.insert( pair<string, int*>( varname, variable ) );
    vector<int> evtVec;
    _hdf_evtVec_int.insert( pair<string, vector<int>>( varname, evtVec ) );
}
void HEPBase::HDF_insert( string varname, float* variable ) {
    _hdf_float.insert( pair<string, float*>( varname, variable ) );
    vector<float> evtVec;
    _hdf_evtVec_float.insert( pair<string, vector<float>>( varname, evtVec ) );
}
void HEPBase::HDF_insert( string varname, double* variable ) {
    _hdf_double.insert( pair<string, double*>( varname, variable ) );
    vector<double> evtVec;
    _hdf_evtVec_double.insert( pair<string, vector<double>>( varname, evtVec ) );
}

void HEPBase::HDF_insert( string varname, bool* variable ) {
    _hdf_bool.insert( pair<string, bool*>( varname, variable ) );
    vector<int> evtVec;
    _hdf_evtVec_bool.insert( pair<string, vector<int>>( varname, evtVec ) );
}
void HEPBase::HDF_insert( string varname, unsigned int* variable ) {
    _hdf_uint.insert( pair<string, unsigned int*>( varname, variable ) );
    vector<int> evtVec;
    _hdf_evtVec_uint.insert( pair<string, vector<int>>( varname, evtVec ) );
}
void HEPBase::HDF_insert( string varname, unsigned char* variable ) {
    _hdf_uchar.insert( pair<string, unsigned char*>( varname, variable ) );
    vector<int> evtVec;
    _hdf_evtVec_uchar.insert( pair<string, vector<int>>( varname, evtVec ) );
}

void HEPBase::HDF_insert( string varname, vector<int>* variable ) {
    _hdf_intVec.insert( pair<string, vector<int>*>( varname, variable ) );
    vector<vector<int>> evtVec;
    _hdf_evtVec_intVec.insert( pair<string, vector<vector<int>>>( varname, evtVec ) );
    _hdf_evtVec_max_size.insert( pair<string, int>( varname, 0 ) );
    _hdf_intVec_N.insert(pair<string,double>(varname, 0) );
    _hdf_intVec_mean.insert(pair<string,double>(varname, 0) );
    _hdf_intVec_std.insert(pair<string,double>(varname, 0) );
}
void HEPBase::HDF_insert( string varname, vector<float>* variable ) {
    _hdf_floatVec.insert( pair<string, vector<float>*>( varname, variable ) );
    vector<vector<float>> evtVec;
    _hdf_evtVec_floatVec.insert( pair<string, vector<vector<float>>>( varname, evtVec ) );
    _hdf_evtVec_max_size.insert( pair<string, int>( varname, 0 ) );
    _hdf_floatVec_N.insert(pair<string,double>(varname, 0) );
    _hdf_floatVec_mean.insert(pair<string,double>(varname, 0) );
    _hdf_floatVec_std.insert(pair<string,double>(varname, 0) );
}
void HEPBase::HDF_insert( string varname, vector<double>* variable ) {
    _hdf_doubleVec.insert( pair<string, vector<double>*>( varname, variable ) );
    vector<vector<double>> evtVec;
    _hdf_evtVec_doubleVec.insert( pair<string, vector<vector<double>>>( varname, evtVec ) );
    _hdf_evtVec_max_size.insert( pair<string, int>( varname, 0 ) );
    _hdf_doubleVec_N.insert(pair<string,double>(varname, 0) );
    _hdf_doubleVec_mean.insert(pair<string,double>(varname, 0) );
    _hdf_doubleVec_std.insert(pair<string,double>(varname, 0) );
}



//---------------------------------------------------------------------------------------------
// Fill HDF maps of event-vectors
//---------------------------------------------------------------------------------------------
void HEPBase::HDF_fill() {
    
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_int.begin(); itr_h != _hdf_evtVec_int.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_int.at((*itr_h).first));
    }
    for( map<string,vector<float>>::iterator itr_h = _hdf_evtVec_float.begin(); itr_h != _hdf_evtVec_float.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_float.at((*itr_h).first));
    }
    for( map<string,vector<double>>::iterator itr_h = _hdf_evtVec_double.begin(); itr_h != _hdf_evtVec_double.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_double.at((*itr_h).first));
    }
    
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_bool.begin(); itr_h != _hdf_evtVec_bool.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_bool.at((*itr_h).first));
    }
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_uint.begin(); itr_h != _hdf_evtVec_uint.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_uint.at((*itr_h).first));
    }
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_uchar.begin(); itr_h != _hdf_evtVec_uchar.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_uchar.at((*itr_h).first));
    }
    
    for( map<string,vector<vector<int>>>::iterator itr_h = _hdf_evtVec_intVec.begin(); itr_h != _hdf_evtVec_intVec.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_intVec.at((*itr_h).first));
        int max_size = _hdf_evtVec_max_size.at((*itr_h).first);
        int vec_size = (*_hdf_intVec.at((*itr_h).first)).size();
        if( vec_size > max_size ) _hdf_evtVec_max_size.at((*itr_h).first) = vec_size;
    }
    for( map<string,vector<vector<float>>>::iterator itr_h = _hdf_evtVec_floatVec.begin(); itr_h != _hdf_evtVec_floatVec.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_floatVec.at((*itr_h).first));
        int max_size = _hdf_evtVec_max_size.at((*itr_h).first);
        int vec_size = (*_hdf_floatVec.at((*itr_h).first)).size();
        if( vec_size > max_size ) _hdf_evtVec_max_size.at((*itr_h).first) = vec_size;
    }
    for( map<string,vector<vector<double>>>::iterator itr_h = _hdf_evtVec_doubleVec.begin(); itr_h != _hdf_evtVec_doubleVec.end(); ++itr_h ) {
        (*itr_h).second.push_back(*_hdf_doubleVec.at((*itr_h).first));
        int max_size = _hdf_evtVec_max_size.at((*itr_h).first);
        int vec_size = (*_hdf_doubleVec.at((*itr_h).first)).size();
        if( vec_size > max_size ) _hdf_evtVec_max_size.at((*itr_h).first) = vec_size;
    }
    
    
}


//---------------------------------------------------------------------------------------------
// Fill HDF maps of event-vectors
//---------------------------------------------------------------------------------------------
void HEPBase::HDF_write() {
    
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_int.begin(); itr_h != _hdf_evtVec_int.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<int>().swap((*itr_h).second);
    }
    for( map<string,vector<float>>::iterator itr_h = _hdf_evtVec_float.begin(); itr_h != _hdf_evtVec_float.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<float>().swap((*itr_h).second);
    }
    for( map<string,vector<double>>::iterator itr_h = _hdf_evtVec_double.begin(); itr_h != _hdf_evtVec_double.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<double>().swap((*itr_h).second);
    }
    
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_bool.begin(); itr_h != _hdf_evtVec_bool.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<int>().swap((*itr_h).second);
    }
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_uint.begin(); itr_h != _hdf_evtVec_uint.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<int>().swap((*itr_h).second);
    }
    for( map<string,vector<int>>::iterator itr_h = _hdf_evtVec_uchar.begin(); itr_h != _hdf_evtVec_uchar.end(); ++itr_h ) {
        _hdf_file->createDataSet("scalars/"+(*itr_h).first, (*itr_h).second);
        vector<int>().swap((*itr_h).second);
    }
    
    for( map<string,vector<vector<int>>>::iterator itr_h = _hdf_evtVec_intVec.begin(); itr_h != _hdf_evtVec_intVec.end(); ++itr_h ) {
        long unsigned int n = (*itr_h).second.size();
        long unsigned int m = _hdf_evtVec_max_size.at((*itr_h).first);
        boost::multi_array<int, 2> boost_array(boost::extents[n][m]);
        for( int in = 0; in < n; ++in ){
            for( int im = 0; im < m; ++im ){
                if( im < (*itr_h).second.at(in).size() ){
                    boost_array[in][im] = (*itr_h).second.at(in).at(im);
                    if( (*itr_h).second.at(in).at(im) > 0 ){
                        _hdf_intVec_N.at((*itr_h).first) += 1.;
                        _hdf_intVec_mean.at((*itr_h).first) += (*itr_h).second.at(in).at(im);
                        _hdf_intVec_std.at((*itr_h).first) += pow((*itr_h).second.at(in).at(im),2);
                    }
                }else{
                    boost_array[in][im] = 0;
                }
            }
        }
        vector<vector<int>>().swap((*itr_h).second);
        _hdf_file->createDataSet("vectors/"+(*itr_h).first, boost_array);
        
        _hdf_intVec_mean.at((*itr_h).first) = _hdf_intVec_mean.at((*itr_h).first)/_hdf_intVec_N.at((*itr_h).first);
        _hdf_intVec_std.at((*itr_h).first) = sqrt(_hdf_intVec_std.at((*itr_h).first)/_hdf_intVec_N.at((*itr_h).first) - pow(_hdf_intVec_mean.at((*itr_h).first),2));
    }
    for( map<string,vector<vector<float>>>::iterator itr_h = _hdf_evtVec_floatVec.begin(); itr_h != _hdf_evtVec_floatVec.end(); ++itr_h ) {
        long unsigned int n = (*itr_h).second.size();
        long unsigned int m = _hdf_evtVec_max_size.at((*itr_h).first);
        boost::multi_array<float, 2> boost_array(boost::extents[n][m]);
        for( int in = 0; in < n; ++in ){
            for( int im = 0; im < m; ++im ){
                if( im < (*itr_h).second.at(in).size() ){
                    boost_array[in][im] = (*itr_h).second.at(in).at(im);
                    if( (*itr_h).second.at(in).at(im) > 0 ){
                        _hdf_floatVec_N.at((*itr_h).first) += 1.;
                        _hdf_floatVec_mean.at((*itr_h).first) += (*itr_h).second.at(in).at(im);
                        _hdf_floatVec_std.at((*itr_h).first) += pow((*itr_h).second.at(in).at(im),2);
                    }
                }else{
                    boost_array[in][im] = 0;
                }
            }
        }
        vector<vector<float>>().swap((*itr_h).second);
        _hdf_file->createDataSet("vectors/"+(*itr_h).first, boost_array);
        
        _hdf_floatVec_mean.at((*itr_h).first) = _hdf_floatVec_mean.at((*itr_h).first)/_hdf_floatVec_N.at((*itr_h).first);
        _hdf_floatVec_std.at((*itr_h).first) = sqrt(_hdf_floatVec_std.at((*itr_h).first)/_hdf_floatVec_N.at((*itr_h).first) - pow(_hdf_floatVec_mean.at((*itr_h).first),2));
    }
    for( map<string,vector<vector<double>>>::iterator itr_h = _hdf_evtVec_doubleVec.begin(); itr_h != _hdf_evtVec_doubleVec.end(); ++itr_h ) {
        long unsigned int n = (*itr_h).second.size();
        long unsigned int m = _hdf_evtVec_max_size.at((*itr_h).first);
        boost::multi_array<double, 2> boost_array(boost::extents[n][m]);
        for( int in = 0; in < n; ++in ){
            for( int im = 0; im < m; ++im ){
                if( im < (*itr_h).second.at(in).size() ){
                    boost_array[in][im] = (*itr_h).second.at(in).at(im);
                    if( (*itr_h).second.at(in).at(im) > 0 ){
                        _hdf_doubleVec_N.at((*itr_h).first) += 1.;
                        _hdf_doubleVec_mean.at((*itr_h).first) += (*itr_h).second.at(in).at(im);
                        _hdf_doubleVec_std.at((*itr_h).first) += pow((*itr_h).second.at(in).at(im),2);
                    }
                }else{
                    boost_array[in][im] = 0;
                }
            }
        }
        vector<vector<double>>().swap((*itr_h).second);
        _hdf_file->createDataSet("vectors/"+(*itr_h).first, boost_array);
        
        _hdf_doubleVec_mean.at((*itr_h).first) = _hdf_doubleVec_mean.at((*itr_h).first)/_hdf_doubleVec_N.at((*itr_h).first);
        _hdf_doubleVec_std.at((*itr_h).first) = sqrt(_hdf_doubleVec_std.at((*itr_h).first)/_hdf_doubleVec_N.at((*itr_h).first) - pow(_hdf_doubleVec_mean.at((*itr_h).first),2));
    }
    
    
    for( map<string,double>::iterator itr_h = _hdf_intVec_N.begin(); itr_h != _hdf_intVec_N.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_N", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_intVec_mean.begin(); itr_h != _hdf_intVec_mean.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_mean", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_intVec_std.begin(); itr_h != _hdf_intVec_std.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_std", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_floatVec_N.begin(); itr_h != _hdf_floatVec_N.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_N", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_floatVec_mean.begin(); itr_h != _hdf_floatVec_mean.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_mean", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_floatVec_std.begin(); itr_h != _hdf_floatVec_std.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_std", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_doubleVec_N.begin(); itr_h != _hdf_doubleVec_N.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_N", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_doubleVec_mean.begin(); itr_h != _hdf_doubleVec_mean.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_mean", (*itr_h).second);
    }
    for( map<string,double>::iterator itr_h = _hdf_doubleVec_std.begin(); itr_h != _hdf_doubleVec_std.end(); ++itr_h ) {
        _hdf_file->createDataSet("metadata/"+(*itr_h).first+"_std", (*itr_h).second);
    }
    
}


//---------------------------------------------------------------------------------------------
// makeSysHist
//---------------------------------------------------------------------------------------------
void HEPBase::makeSysHist( string nametitle, int nbins, double xmin, double xmax, string xtitle, string ytitle, string drawOption, double xAxisOffset, double yAxisOffset ) {
    
    vector<int> sysRegions;
    if( sys_regions.size() > 0 ){
        for( int ireg = 0; ireg < sys_regions.size(); ++ireg ){
            sysRegions.push_back(sys_regions.at(ireg));
        }
    }else{
        sysRegions.push_back(0);
    }
    
    for( int ireg = 0; ireg < sysRegions.size(); ++ireg ){
        string universetitle = nametitle + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysID_lateral) + "_" + to_string(_Universe);    
        TH1D hist(universetitle.c_str(), universetitle.c_str(), nbins, xmin, xmax );
        hist.GetXaxis()->SetTitle( xtitle.c_str() );
        hist.GetYaxis()->SetTitle( ytitle.c_str() );
        hist.GetYaxis()->SetTitleOffset( yAxisOffset );
        hist.GetXaxis()->SetTitleOffset( xAxisOffset );
        hist.Sumw2();
        _systematics1D.insert( pair<string, TH1D>( universetitle, hist ) );
        _systematics1DDrawOptions.insert( pair<string,string>( universetitle, drawOption ) );
    }
    
    if( _sysID_lateral == 0 ) {
        for( int ireg = 0; ireg < sysRegions.size(); ++ireg ){
            for( int ivert = 0; ivert < _sysIDs_vertical.size(); ++ivert ){
                for( int iuniv = 0; iuniv < sys_vertical_size.at(ivert); ++iuniv ){
                    string universetitle = nametitle + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysIDs_vertical.at(ivert)) + "_" + to_string(iuniv);    
                    TH1D hist(universetitle.c_str(), universetitle.c_str(), nbins, xmin, xmax );
                    hist.GetXaxis()->SetTitle( xtitle.c_str() );
                    hist.GetYaxis()->SetTitle( ytitle.c_str() );
                    hist.GetYaxis()->SetTitleOffset( yAxisOffset );
                    hist.GetXaxis()->SetTitleOffset( xAxisOffset );
                    hist.Sumw2();
                    _systematics1D.insert( pair<string, TH1D>( universetitle, hist ) );
                    _systematics1DDrawOptions.insert( pair<string,string>( universetitle, drawOption ) );
                }
            }
        }
    }
}


void HEPBase::makeSysHist( string nametitle, int nbinsx, double xmin, double xmax, int nbinsy, double ymin, double ymax, string xtitle, string ytitle, string ztitle, string drawOption, double xAxisOffset, double yAxisOffset, double zAxisOffset ) {
    
    vector<int> sysRegions;
    if( sys_regions.size() > 0 ){
        for( int ireg = 0; ireg < sys_regions.size(); ++ireg ){
            sysRegions.push_back(sys_regions.at(ireg));
        }
    }else{
        sysRegions.push_back(0);
    }
    
    for( int ireg = 0; ireg < sysRegions.size(); ++ireg ){
        string universetitle = nametitle + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysID_lateral) + "_" + to_string(_Universe); 
        TH2D hist(universetitle.c_str(), universetitle.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax );
        hist.GetXaxis()->SetTitle( xtitle.c_str() );
        hist.GetYaxis()->SetTitle( ytitle.c_str() );
        hist.GetZaxis()->SetTitle( ztitle.c_str() );
        hist.GetXaxis()->SetTitleOffset( xAxisOffset );
        hist.GetYaxis()->SetTitleOffset( yAxisOffset );
        hist.GetZaxis()->SetTitleOffset( zAxisOffset );
        hist.Sumw2();
        _systematics2D.insert( pair<string, TH2D>( universetitle, hist ) );
        _systematics2DDrawOptions.insert( pair<string,string>( universetitle, drawOption ) );
    }
    
    if( _sysID_lateral == 0 ) {
        for( int ireg = 0; ireg < sysRegions.size(); ++ireg ){
            for( int ivert = 0; ivert < _sysIDs_vertical.size(); ++ivert ){
                for( int iuniv = 0; iuniv < sys_vertical_size.at(ivert); ++iuniv ){
                    string universetitle = nametitle + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysIDs_vertical.at(ivert)) + "_" + to_string(iuniv);    
                    TH2D hist(universetitle.c_str(), universetitle.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax );
                    hist.GetXaxis()->SetTitle( xtitle.c_str() );
                    hist.GetYaxis()->SetTitle( ytitle.c_str() );
                    hist.GetZaxis()->SetTitle( ztitle.c_str() );
                    hist.GetXaxis()->SetTitleOffset( xAxisOffset );
                    hist.GetYaxis()->SetTitleOffset( yAxisOffset );
                    hist.GetZaxis()->SetTitleOffset( zAxisOffset );
                    hist.Sumw2();
                    _systematics2D.insert( pair<string, TH2D>( universetitle, hist ) );
                    _systematics2DDrawOptions.insert( pair<string,string>( universetitle, drawOption ) );
                }
            }
        }
    }
}


//---------------------------------------------------------------------------------------------
// FillSystematic
//---------------------------------------------------------------------------------------------
void HEPBase::FillSystematic( string varName, double varEntry, double evtWeight ){
    
    vector<int> sysRegions;
    vector<bool> regionFlags;
    if( sys_regions.size() > 0 ){
        for( int ireg = 0; ireg < sys_regions.size(); ++ireg ){
            sysRegions.push_back(sys_regions.at(ireg));
            regionFlags.push_back(RegionID == sys_regions.at(ireg));
        }
    }else{
        sysRegions.push_back(0);
        regionFlags.push_back(true);
    }
    
        
    for( int ireg = 0; ireg < regionFlags.size(); ++ireg ){
        if( regionFlags.at(ireg) ){
            string varSys = varName + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysID_lateral) + "_" + to_string(_Universe); 
            _systematics1D.at(varSys.c_str()).Fill( varEntry, evtWeight );
        }
    }
        
    if( _sysID_lateral == 0 ) {
        for( int ireg = 0; ireg < regionFlags.size(); ++ireg ){
            if( regionFlags.at(ireg) ){
                for( int ivert = 0; ivert < _sysIDs_vertical.size(); ++ivert ){
                    string sysName = _sysNames_vertical.at(ivert);
                    for( int iuniv = 0; iuniv < sys_vertical_size.at(ivert); ++iuniv ){
                        string varSys = varName + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysIDs_vertical.at(ivert)) + "_" + to_string(iuniv); 
                        float sysWeight = evtWeight*sys_vertical_sfs.at(sysName).at(iuniv);
                        _systematics1D.at(varSys.c_str()).Fill( varEntry, sysWeight );
                    }
                }
            }
        }
    }
}


void HEPBase::FillSystematic( string varName, double xvarEntry, double yvarEntry, double evtWeight ){
    
    vector<int> sysRegions;
    vector<bool> regionFlags;
    if( sys_regions.size() > 0 ){
        for( int ireg = 0; ireg < sys_regions.size(); ++ireg ){
            sysRegions.push_back(sys_regions.at(ireg));
            regionFlags.push_back(RegionID == sys_regions.at(ireg));
        }
    }else{
        sysRegions.push_back(0);
        regionFlags.push_back(true);
    }
    
        
    for( int ireg = 0; ireg < regionFlags.size(); ++ireg ){
        if( regionFlags.at(ireg) ){
            string varSys = varName + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysID_lateral) + "_" + to_string(_Universe); 
            _systematics2D.at(varSys.c_str()).Fill( xvarEntry, yvarEntry, evtWeight );
        }
    }
        
    if( _sysID_lateral == 0 ) {
        for( int ireg = 0; ireg < regionFlags.size(); ++ireg ){
            if( regionFlags.at(ireg) ){
                for( int ivert = 0; ivert < _sysIDs_vertical.size(); ++ivert ){
                    string sysName = _sysNames_vertical.at(ivert);
                    for( int iuniv = 0; iuniv < sys_vertical_size.at(ivert); ++iuniv ){
                        string varSys = varName + "_" + to_string(sysRegions.at(ireg)) + "_" + to_string(_sysIDs_vertical.at(ivert)) + "_" + to_string(iuniv); 
                        float sysWeight = evtWeight*sys_vertical_sfs.at(sysName).at(iuniv);
                        _systematics2D.at(varSys.c_str()).Fill( xvarEntry, yvarEntry, sysWeight );
                    }
                }
            }
        }
    }
}


//---------------------------------------------------------------------------------------------
// makeHist
//---------------------------------------------------------------------------------------------
void HEPBase::makeHist( string nametitle, int nbinsx, double xmin, double xmax, int nbinsy, double ymin, double ymax, string xtitle, string ytitle, string ztitle, string drawOption, double xAxisOffset, double yAxisOffset, double zAxisOffset ) {

    TH2D hist(nametitle.c_str(), nametitle.c_str(), nbinsx, xmin, xmax, nbinsy, ymin, ymax );
    hist.GetXaxis()->SetTitle( xtitle.c_str() );
    hist.GetYaxis()->SetTitle( ytitle.c_str() );
    hist.GetZaxis()->SetTitle( ztitle.c_str() );
    hist.GetXaxis()->SetTitleOffset( xAxisOffset );
    hist.GetYaxis()->SetTitleOffset( yAxisOffset );
    hist.GetZaxis()->SetTitleOffset( zAxisOffset );
    _histograms2D.insert( pair<string, TH2D>( nametitle, hist ) );
    _histograms2DDrawOptions.insert( pair<string,string>( nametitle, drawOption ) );

}

void HEPBase::makeHist( string nametitle, int nbins, double xmin, double xmax, string xtitle, string ytitle, string drawOption, double xAxisOffset, double yAxisOffset ) {

    TH1D hist(nametitle.c_str(), nametitle.c_str(), nbins, xmin, xmax );
    hist.GetXaxis()->SetTitle( xtitle.c_str() );
    hist.GetYaxis()->SetTitle( ytitle.c_str() );
    hist.GetYaxis()->SetTitleOffset( yAxisOffset );
    hist.GetXaxis()->SetTitleOffset( xAxisOffset );
    _histograms1D.insert( pair<string, TH1D>( nametitle, hist ) );
    _histograms1DDrawOptions.insert( pair<string,string>( nametitle, drawOption ) );
}


//---------------------------------------------------------------------------------------------
// MakeEfficiencyPlot
//---------------------------------------------------------------------------------------------
void HEPBase::MakeEfficiencyPlot( TH1D hpass, TH1D htotal, string triggerName ) {
    TCanvas c("","");
    TGraphAsymmErrors geff;
    geff.BayesDivide( &hpass, &htotal );
    geff.GetXaxis()->SetTitle( hpass.GetXaxis()->GetTitle() );
    string ytitle = "#varepsilon (" + triggerName + ")";
    geff.GetYaxis()->SetTitle( ytitle.c_str() );
    string efftitle = _outputDirectory + "efficiency_" + triggerName;
    geff.SetNameTitle(efftitle.c_str(), efftitle.c_str());
    geff.SetMarkerColor(kBlue);
    geff.Draw("APZ"); 
    for( vector<string>::iterator itr_f = _plotFormats.begin(); itr_f != _plotFormats.end(); ++itr_f ) {
        string thisPlotName = efftitle + (*itr_f);
        c.Print( thisPlotName.c_str() );
    }

}


//---------------------------------------------------------------------------------------------
// setStyle
//---------------------------------------------------------------------------------------------
void HEPBase::setStyle( double ytoff, bool marker, double left_margin ) {
// use plain black on white colors
Int_t icol=0;
gStyle->SetFrameBorderMode(icol);
gStyle->SetCanvasBorderMode(icol);
gStyle->SetPadBorderMode(icol);
gStyle->SetPadColor(icol);
gStyle->SetCanvasColor(icol);
gStyle->SetStatColor(icol);
gStyle->SetTitleFillColor(icol);
// set the paper & margin sizes
gStyle->SetPaperSize(20,26);
gStyle->SetPadTopMargin(0.10);
gStyle->SetPadRightMargin(0.15);
gStyle->SetPadBottomMargin(0.16);
gStyle->SetPadLeftMargin(0.15);
// use large fonts
Int_t font=62;
Double_t tsize=0.04;
gStyle->SetTextFont(font);
gStyle->SetTextSize(tsize);
gStyle->SetLabelFont(font,"x");
gStyle->SetTitleFont(font,"x");
gStyle->SetLabelFont(font,"y");
gStyle->SetTitleFont(font,"y");
gStyle->SetLabelFont(font,"z");
gStyle->SetTitleFont(font,"z");
gStyle->SetLabelSize(tsize,"x");
gStyle->SetTitleSize(tsize,"x");
gStyle->SetLabelSize(tsize,"y");
gStyle->SetTitleSize(tsize,"y");
gStyle->SetLabelSize(tsize,"z");
gStyle->SetTitleSize(tsize,"z");
gStyle->SetTitleBorderSize(0);
//use bold lines and markers
if ( marker ) {
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
}
gStyle->SetHistLineWidth(Width_t(3.));
// postscript dashes
gStyle->SetLineStyleString(2,"[12 12]");
gStyle->SetOptStat(0);
gStyle->SetOptFit(1111);
// put tick marks on top and RHS of plots
gStyle->SetPadTickX(1);
gStyle->SetPadTickY(1);
// DLA overrides
gStyle->SetPadLeftMargin(left_margin);
gStyle->SetPadBottomMargin(0.13);
gStyle->SetTitleYOffset(ytoff);
gStyle->SetTitleXOffset(1.0);
gStyle->SetOptTitle(0);
//gStyle->SetStatStyle(0);
//gStyle->SetStatFontSize();
gStyle->SetStatW(0.17);

}



