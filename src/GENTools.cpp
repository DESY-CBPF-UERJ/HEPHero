#include "HEPBase.h"

//---------------------------------------------------------------------------------------------
// Calculate Gen Variables
//---------------------------------------------------------------------------------------------
void HEPBase::calculate_gen_variables(){
    
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


//---------------------------------------------------------------------------------------------
// Compute particle mass from stable particles at the end of the chain
//---------------------------------------------------------------------------------------------
float HEPBase::part_mass( int barcode ){
    
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


//---------------------------------------------------------------------------------------------
// PLOT GRAPHS OF THE EVENTS IN THE LIST
//---------------------------------------------------------------------------------------------
void HEPBase::plot_events(vector<int> events){
    
    for( unsigned int ievt = 0; ievt < events.size(); ++ievt ){
        if( _EventPosition == events[ievt] ){
            _dot_writer = new HepMC3::WriterDOT(_outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".dot", _ascii_file->run_info());
            _dot_writer->write_event(_evt);
            string dot_command = "dot -Tpng " + _outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".dot > " + _outputDirectory + "/Views/event_" + to_string(_EventPosition) + ".png ";
            system(dot_command.c_str());
        }
    }
}

  
//---------------------------------------------------------------------------------------------
// PRINT INFO ABOUT THE SELECTION PROCESS
//---------------------------------------------------------------------------------------------
void HEPBase::WriteGenCutflowInfo(){
    
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



//---------------------------------------------------------------------------------------------
// GENERATOR ROUTINES INSIDE THE LOOP
//---------------------------------------------------------------------------------------------
bool HEPBase::GenRoutines(){

    //======HEPHeroGEN===============================================================
    _evt.set_units(HepMC3::Units::GEV, HepMC3::Units::MM);

    weights_value = _evt.weights();
    if( _EventPosition == 0 ) weights_name = _ascii_file->run_info()->weight_names();

    // PROCESS INFO
    if( _EventPosition == 0 ){
        if( _has_pdf ){
            std::shared_ptr<HepMC3::GenPdfInfo> pdf = _evt.attribute<HepMC3::GenPdfInfo>("GenPdfInfo");
            pdf_id1 = pdf->pdf_id[0];
            pdf_id2 = pdf->pdf_id[1];
        }else{
            pdf_id1 = -1;
            pdf_id2 = -1;
        }

        _N_PS_weights = _evt.weights().size();

        momentum_unit_id = _evt.momentum_unit();
        length_unit_id = _evt.length_unit();
        _momentum_unit = "MeV";
        _length_unit = "mm";
        if( momentum_unit_id == 1 ) _momentum_unit = "GeV";
        if( length_unit_id == 1 ) _length_unit = "cm";

    }

    if( _EventPosition+1 == _NumberEntries ){
        if( _has_xsec ){
            std::shared_ptr<HepMC3::GenCrossSection> cs = _evt.attribute<HepMC3::GenCrossSection>("GenCrossSection");
            cross_section = cs->xsec("Default");
            cross_section_unc = cs->xsec_err("Default");
        }else{
            cross_section = -1;
            cross_section_unc = -1;
        }
    }

    // EVENT INFO
    event_number = _evt.event_number();

    if( _has_pdf ){
        std::shared_ptr<HepMC3::GenPdfInfo> pdf = _evt.attribute<HepMC3::GenPdfInfo>("GenPdfInfo");
        scalePDF = pdf->scale;      // Q-scale used in evaluation of PDF’s (in GeV)
        id1 = pdf->parton_id[0];    // flavour code of first parton
        id2 = pdf->parton_id[1];    // flavour code of second parton
        x1 = pdf->x[0];             // fraction of beam momentum carried by first parton (”beam side”)
        x2 = pdf->x[1];             // fraction of beam momentum carried by second parton (”target side”)
        pdf1 = pdf->xf[0];          // PDF (id1, x1, Q) This should be of the form x*f(x)
        pdf2 = pdf->xf[1];          // PDF (id2, x2, Q) This should be of the form x*f(x)
    }

    return true;

}



