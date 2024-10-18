#!/bin/bash

ANALYSIS=$(head -n 1 tools/analysis.txt)

if [ ! -e ${ANALYSIS}/ana/${1}.cpp ]; then
  echo "#include \"HEPHero.h\"" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Description:" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Define output variables" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "namespace ${1}{" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //int variable1Name;   [example]" >> ${ANALYSIS}/ana/${1}.cpp
  echo "}" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Define output derivatives" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "void HEPHero::Setup${1}() {" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======SETUP CUTFLOW==========================================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //_cutFlow.insert(pair<string,double>("CutName", 0) );   [example]'>> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======SETUP HISTOGRAMS=======================================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //makeHist( "histogram1DName", 40, 0., 40., "xlabel", "ylabel" );   [example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //makeHist( "histogram2DName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======SETUP SYSTEMATIC HISTOGRAMS============================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //sys_regions = { 0, 1, 2 }; [example] // Choose regions as defined in RegionID. Empty vector means that all events will be used.' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //makeSysHist( "histogram1DSysName", 40, 0., 40., "xlabel", "ylabel" );   [example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //makeSysHist( "histogram2DSysName", 40, 0., 40., 100, 0., 50., "xlabel",  "ylabel", "zlabel", "COLZ" );   [example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======SETUP OUTPUT BRANCHES==================================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //_outputTree->Branch(\"variable1NameInTheTree\", &${1}::variable1Name );  [example]" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======SETUP INFORMATION IN OUTPUT HDF5 FILE==================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //HDF_insert(\"variable1NameInTheTree\", &${1}::variable1Name );  [example]" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    return;">> ${ANALYSIS}/ana/${1}.cpp
  echo "}">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Define the selection region" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "bool HEPHero::${1}Region() {">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //-------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "    // Cut description" >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //-------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //if( !(CutCondition) ) return false;           [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //_cutFlow.at("CutName") += evtWeight;          [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    return true;">> ${ANALYSIS}/ana/${1}.cpp
  echo "}">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Write your analysis code here" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "void HEPHero::${1}Selection() {">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======ASSIGN VALUES TO THE OUTPUT VARIABLES==================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //${1}::variable1Name = 100;      [Example]" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======FILL THE HISTOGRAMS====================================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //_histograms1D.at("histogram1DName").Fill( var, evtWeight );               [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //_histograms2D.at("histogram2DName").Fill( var1, var2, evtWeight );        [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======FILL THE OUTPUT TREE===================================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //_outputTree->Fill();' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    //======FILL THE OUTPUT HDF5 INFO===============================================================" >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //HDF_fill();' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    return;">> ${ANALYSIS}/ana/${1}.cpp
  echo "}">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Produce systematic histograms" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "void HEPHero::${1}Systematic() {">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //FillSystematic( "histogram1DSysName", var, evtWeight );  [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //FillSystematic( "histogram2DSysName", var1, var2, evtWeight );  [Example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo "}">> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "// Make efficiency plots" >> ${ANALYSIS}/ana/${1}.cpp
  echo "//-------------------------------------------------------------------------------------------------" >> ${ANALYSIS}/ana/${1}.cpp
  echo "void HEPHero::Finish${1}() {" >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo '    //MakeEfficiencyPlot( _histograms1D.at("Matched_pt"), _histograms1D.at("all_pt"), "Match_pt" );   [example]' >> ${ANALYSIS}/ana/${1}.cpp
  echo  >> ${ANALYSIS}/ana/${1}.cpp
  echo "    return;">> ${ANALYSIS}/ana/${1}.cpp
  echo "}">> ${ANALYSIS}/ana/${1}.cpp
else
  echo "SOURCE FILE ALREADY EXISTS. NOT CREATING A NEW SOURCE FILE FOR Region ${1}"
fi


if ! grep -q "Setup${1}()" ${ANALYSIS}/include/HEPHero.h; then
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void Setup'${1}'();' ${ANALYSIS}/include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        bool '${1}'Region();' ${ANALYSIS}/include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void '${1}'Selection();' ${ANALYSIS}/include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void '${1}'Systematic();' ${ANALYSIS}/include/HEPHero.h
  sed -i '/INSERT YOUR SELECTION HERE/ i\        void Finish'${1}'();' ${ANALYSIS}/include/HEPHero.h
else
  echo "Setup${1} ALREADY KNOW TO HEPHero.h. LEAVING FILE UNCHANGED"
fi


if ! grep -q "Setup${1}()" ${ANALYSIS}/src/HEPHeroB.cpp; then
  sed -i '/SETUP YOUR SELECTION HERE/ i\    else if( _SELECTION == "'${1}'" ) Setup'${1}'();' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/SET THE REGION OF YOUR SELECTION HERE/ i\    if( _SELECTION == "'${1}'" && !'${1}'Region() ) Selected = false;' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/CALL YOUR SELECTION HERE/ i\    if( _SELECTION == "'${1}'" ) '${1}'Selection();' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/PRODUCE THE SYSTEMATIC OF YOUR SELECTION HERE/ i\    if( _SELECTION == "'${1}'" ) '${1}'Systematic();' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/FINISH YOUR SELECTION HERE/ i\    if( _SELECTION == "'${1}'" ) Finish'${1}'();' ${ANALYSIS}/src/HEPHeroB.cpp
else
  echo "Setup${1} ALREADY KNOW TO HEPHero.cpp. LEAVING FILE UNCHANGED"
fi


if ! grep -q "\${ANALYSIS}/ana/${1}.cpp" ${ANALYSIS}/ana/CMakeLists.txt; then
  sed -i "s~PRIVATE~PRIVATE \${ANALYSIS}/ana/${1}.cpp~" ${ANALYSIS}/ana/CMakeLists.txt
else
  echo "${1}.cpp ALREADY KNOWN TO CMakeLists.txt. LEAVING FILE UNCHANGED"
fi




