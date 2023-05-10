#include "HEPHero.h"

//---------------------------------------------------------------------------------------------------------------
// Calculate Gen Variables
//---------------------------------------------------------------------------------------------------------------
void HEPHero::calculate_gen_variables(){
    
    GEN_HT = 0;
    TLorentzVector MHT;
    TLorentzVector MET;
    MHT.SetPtEtaPhiE(0, 0, 0, 0);
    MET.SetPtEtaPhiE(0, 0, 0, 0);
    for (auto p: _evt.particles()) {
        if( !(*p).end_vertex() || ((*p).status() == 1) ){
            
            bool jet_component = false;
            bool has_Boson = false;
            for (auto ancestor : HepMC3::Relatives::ANCESTORS(p) ) {
                //if( (*ancestor).pdg_id() == 81 ){
                if( (abs((*ancestor).pdg_id()) <= 6) || ((*ancestor).pdg_id() == 21) ){
                    jet_component = true;
                    //break;
                }
                if( ((*ancestor).pdg_id() >= 22) && ((*ancestor).pdg_id() <= 37) ){
                    has_Boson = true;
                }
            }
            
            if( has_Boson ) jet_component = false;
            
            TLorentzVector part;
            part.SetPtEtaPhiE((*p).momentum().perp(), (*p).momentum().eta(), (*p).momentum().phi(), (*p).momentum().e());
            
            if( jet_component ){ 
                MHT = MHT - part;
                GEN_HT += (*p).momentum().perp();
            }
            
            if( (abs((*p).pdg_id()) == 12) || (abs((*p).pdg_id()) == 14) || (abs((*p).pdg_id()) == 16) || (abs((*p).pdg_id()) == 18) || (abs((*p).pdg_id()) == 1000022) || (abs((*p).pdg_id()) == 1000023) || (abs((*p).pdg_id()) == 1000025) || (abs((*p).pdg_id()) == 1000035) ){
                MET = MET + part;
            }
        }
    }
    GEN_MET_pt = MET.Pt();
    GEN_MET_phi = MET.Phi();
    GEN_MHT_pt = MHT.Pt();
    GEN_MHT_phi = MHT.Phi();
}


//---------------------------------------------------------------------------------------------------------------
// Compute particle mass from stable particles at the end of the chain
//---------------------------------------------------------------------------------------------------------------
float HEPHero::part_mass( int barcode ){
    
    /*
    TLorentzVector part;
    int N_descendant = 0;
    for (auto p: _evt.particles()) {
        if( !(*p).end_vertex() || ((*p).status() == 1) ){
            
            bool is_descendant = false;
            for (auto ancestor : HepMC3::Relatives::ANCESTORS(p) ) {
                if( (*ancestor).barcode() == barcode ){
                    is_descendant = true;
                    break;
                }
            }
            
            if( is_descendant ){
                if( N_descendant == 0 ){
                    part.SetPtEtaPhiE((*p).momentum().perp(), (*p).momentum().eta(), (*p).momentum().phi(), (*p).momentum().e());
                }else{
                    TLorentzVector desc_part;
                    desc_part.SetPtEtaPhiE((*p).momentum().perp(), (*p).momentum().eta(), (*p).momentum().phi(), (*p).momentum().e());
                    part = part + desc_part;
                }
                N_descendant += 1;
            }
        }
    }
    
    return part.M();
    */
    return 0.0;
}


//---------------------------------------------------------------------------------------------------------------
// PLOT GRAPHS OF THE EVENTS IN THE LIST
//---------------------------------------------------------------------------------------------------------------
void HEPHero::plot_events(vector<int> events){
    
    for( unsigned int ievt = 0; ievt < events.size(); ++ievt ){
        if( _EventPosition == events[ievt] ){
            _dot_writer = new HepMC3::WriterDOT(_outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".dot", _ascii_file->run_info());
            _dot_writer->write_event(_evt);
            string dot_command = "dot -Tpng " + _outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".dot > " + _outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".png ";
            system(dot_command.c_str());
        }
    }
}

  
//---------------------------------------------------------------------------------------------------------------
// PRINT INFO ABOUT THE SELECTION PROCESS
//---------------------------------------------------------------------------------------------------------------
void HEPHero::WriteGenCutflowInfo(){
    
    if( _sysID_lateral == 0 ) {
        _CutflowFile.open( _CutflowFileName.c_str(), ios::app );
        _CutflowFile << "-----------------------------------------------------------------------------------" << endl;
        _CutflowFile << "LHAPDF set id1: " << pdf_id1 << endl;
        _CutflowFile << "LHAPDF set id2: " << pdf_id2 << endl;
        _CutflowFile << "Cross section: " << cross_section << " +- " << cross_section_unc << " pb" << endl;
        _CutflowFile << "Number of weights: " << _N_PS_weights << endl;
        _CutflowFile << "Momentum unit: " << _momentum_unit << endl;
        _CutflowFile << "Length unit: " << _length_unit << endl;
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


//---------------------------------------------------------------------------------------------------------------
// Weight corrections
//---------------------------------------------------------------------------------------------------------------
/*
void HEPHero::Weight_corrections(){
    
    //pileup_wgt = 1.;
    //electron_wgt = 1.;
    
    if(dataset_group != "Data"){
        
        
        if( apply_pileup_wgt ){
            pileup_wgt = pileup_corr->evaluate({Pileup_nTrueInt, "nominal"});
            evtWeight *= pileup_wgt;
        }
        
        if( apply_electron_wgt ){
            electron_wgt = GetElectronWeight("cv");
            evtWeight *= electron_wgt;
        }
        
    }
    
}
*/


//---------------------------------------------------------------------------------------------------------------
// Vertical systematics
// Keep the same order used in runSelection.py
//--------------------------------------------------------------------------------------------------------------- 
/*
void HEPHero::VerticalSys(){
    if( _sysID_lateral == 0 ) {   
        sys_vertical_sfs.clear();
        
        int ivert2 = 1;
        for( int ivert = 0; ivert < (_N_PS_weights-1)/2.; ++ivert ){
            
            vector<float> sfs;
            sfs.push_back(weights_value[ivert2]);      // DOWN
            sfs.push_back(weights_value[ivert2+1]);    // UP
            sys_vertical_sfs.insert(pair<string, vector<float>>(_sysNames_vertical[ivert], sfs));
            ivert2 += 2;
        }
    }
}
*/





