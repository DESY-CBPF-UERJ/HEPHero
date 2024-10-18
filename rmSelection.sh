#!/bin/bash

ANALYSIS=$(head -n 1 tools/analysis.txt)

if grep -q "Setup${1}()" ${ANALYSIS}/include/HEPHero.h; then
  sed -i '/void Setup'${1}'();/d' ${ANALYSIS}/include/HEPHero.h
  sed -i '/bool '${1}'Region();/d' ${ANALYSIS}/include/HEPHero.h
  sed -i '/void '${1}'Selection();/d' ${ANALYSIS}/include/HEPHero.h
  sed -i '/void '${1}'Systematic();/d' ${ANALYSIS}/include/HEPHero.h
  sed -i '/void Finish'${1}'();/d' ${ANALYSIS}/include/HEPHero.h
else
  echo "Setup${1} DOESN'T EXIST IN HEPHero.h. LEAVING FILE UNCHANGED"
fi

if grep -q "Setup${1}()" ${ANALYSIS}/src/HEPHeroB.cpp; then
  sed -i '/if( _SELECTION == "'${1}'" ) Setup'${1}'();/d' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/if( _SELECTION == "'${1}'" && !'${1}'Region() ) Selected = false;/d' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) '${1}'Selection();/d' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) '${1}'Systematic();/d' ${ANALYSIS}/src/HEPHeroB.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) Finish'${1}'();/d' ${ANALYSIS}/src/HEPHeroB.cpp
else
  echo "Setup${1} DOESN'T EXIST IN HEPHero.cpp. LEAVING FILE UNCHANGED"
fi


if grep -q "\${ANALYSIS}/ana/${1}.cpp" ${ANALYSIS}/ana/CMakeLists.txt; then
  sed -i "s~ \${ANALYSIS}/ana/${1}.cpp~~" ${ANALYSIS}/ana/CMakeLists.txt
else
  echo "${1}.cpp DOESN'T EXIST IN CMakeLists.txt. LEAVING FILE UNCHANGED"
fi

#rm ana${1}.o
