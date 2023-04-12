#!/bin/bash

if [ ! -e ana/${1}.cpp ]; then 
  echo "#include \"HEPHero.h\"" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Description:" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Define output variables" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "namespace ${1}{" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //int variable1Name;   [example]" >> ana/${1}.cpp
  echo "}" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Define output derivatives" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "void HEPHero::Setup${1}() {" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======SETUP CUTFLOW==========================================================================" >> ana/${1}.cpp
  echo '    //_cutFlow.insert(pair<string,double>("CutName", 0) );   [example]'>> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======SETUP HISTOGRAMS=======================================================================" >> ana/${1}.cpp
  echo '    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]' >> ana/${1}.cpp
  echo '    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======SETUP SYSTEMATIC HISTOGRAMS============================================================" >> ana/${1}.cpp
  echo '    //sys_regions = { 0, 1, 2 }; [example] // Choose regions as defined in RegionID. Empty vector means that all events will be used.' >> ana/${1}.cpp
  echo '    //makeSysHist( "histogram1DSysName", 40, 0., 40., "xlabel", "ylabel" );   [example]' >> ana/${1}.cpp
  echo '    //makeSysHist( "histogram2DSysName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======SETUP OUTPUT BRANCHES==================================================================" >> ana/${1}.cpp
  echo "    //_outputTree->Branch(\"variable1NameInTheTree\", &${1}::variable1Name );  [example]" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======SETUP INFORMATION IN OUTPUT HDF5 FILE==================================================" >> ana/${1}.cpp
  echo "    //HDF_insert(\"variable1NameInTheTree\", &${1}::variable1Name );  [example]" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    return;">> ana/${1}.cpp
  echo "}">> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Define the selection region" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "bool HEPHero::${1}Region() {">> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //-------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "    // Cut description" >> ana/${1}.cpp
  echo "    //-------------------------------------------------------------------------" >> ana/${1}.cpp
  echo '    //if( !(CutCondition) ) return false;           [Example]' >> ana/${1}.cpp
  echo '    //_cutFlow.at("CutName") += evtWeight;          [Example]' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    return true;">> ana/${1}.cpp
  echo "}">> ana/${1}.cpp  
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Write your analysis code here" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "void HEPHero::${1}Selection() {">> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================" >> ana/${1}.cpp
  echo "    //${1}::variable1Name = 100;      [Example]" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======FILL THE HISTOGRAMS====================================================================" >> ana/${1}.cpp
  echo '    //_histograms1D.at("histogram1DName").Fill( var, evtWeight );               [Example]' >> ana/${1}.cpp
  echo '    //_histograms2D.at("histogram2DName").Fill( var1, var2, evtWeight );        [Example]' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======FILL THE OUTPUT TREE===================================================================" >> ana/${1}.cpp
  echo '    //_outputTree->Fill();' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    //======FILL THE OUTPUT HDF5 INFO===============================================================" >> ana/${1}.cpp
  echo '    //HDF_fill();' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    return;">> ana/${1}.cpp
  echo "}">> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Produce systematic histograms" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "void HEPHero::${1}Systematic() {">> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo '    //FillSystematic( "histogram1DSysName", var, evtWeight );  [Example]' >> ana/${1}.cpp
  echo '    //FillSystematic( "histogram2DSysName", var1, var2, evtWeight );  [Example]' >> ana/${1}.cpp
  echo "}">> ana/${1}.cpp  
  echo  >> ana/${1}.cpp
  echo  >> ana/${1}.cpp  
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "// Make efficiency plots" >> ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ana/${1}.cpp
  echo "void HEPHero::Finish${1}() {" >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo '    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]' >> ana/${1}.cpp
  echo  >> ana/${1}.cpp
  echo "    return;">> ana/${1}.cpp
  echo "}">> ana/${1}.cpp
else
  echo "SOURCE FILE ALREADY EXISTS. NOT CREATING A NEW SOURCE FILE FOR Region ${1}"
fi




if ! grep -q "Setup${1}()" include/HEPHero.h; then
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void Setup'${1}'();' include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        bool '${1}'Region();' include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void '${1}'Selection();' include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void '${1}'Systematic();' include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void Finish'${1}'();' include/HEPHero.h
else
  echo "Setup${1} ALREADY KNOW TO HEPHero.h. LEAVING FILE UNCHANGED"
fi

if ! grep -q "Setup${1}()" src/HEPHero.cpp; then
  sed -i '/SETUP YOUR SELECTION HERE/ i\    else if( _SELECTION == "'${1}'" ) Setup'${1}'();' src/HEPHero.cpp
  sed -i '/SET THE REGION OF YOUR SELECTION HERE/ i\        if( _SELECTION == "'${1}'" && !'${1}'Region() ) Selected = false;' src/HEPHero.cpp
  sed -i '/CALL YOUR SELECTION HERE/ i\            if( _SELECTION == "'${1}'" ) '${1}'Selection();' src/HEPHero.cpp
  sed -i '/PRODUCE THE SYSTEMATIC OF YOUR SELECTION HERE/ i\        if( _SELECTION == "'${1}'" ) '${1}'Systematic();' src/HEPHero.cpp
  sed -i '/FINISH YOUR SELECTION HERE/ i\    if( _SELECTION == "'${1}'" ) Finish'${1}'();' src/HEPHero.cpp
else
  echo "Setup${1} ALREADY KNOW TO HEPHero.cpp. LEAVING FILE UNCHANGED"
fi


if ! grep -q "ana/${1}.cpp" CMakeLists.txt; then 
  sed -i "s~HEPObjects.cpp~HEPObjects.cpp ana/${1}.cpp~" CMakeLists.txt
else
  echo "${1}.cpp ALREADY KNOWN TO CMakeLists.txt. LEAVING FILE UNCHANGED"
fi




