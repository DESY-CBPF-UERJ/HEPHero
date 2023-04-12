#!/bin/bash

if grep -q "Setup${1}()" include/HEPHero.h; then
  sed -i '/void Setup'${1}'();/d' include/HEPHero.h
  sed -i '/bool '${1}'Region();/d' include/HEPHero.h
  sed -i '/void '${1}'Selection();/d' include/HEPHero.h
  sed -i '/void '${1}'Systematic();/d' include/HEPHero.h
  sed -i '/void Finish'${1}'();/d' include/HEPHero.h
else
  echo "Setup${1} DOESN'T EXIST IN HEPHero.h. LEAVING FILE UNCHANGED"
fi

if grep -q "Setup${1}()" src/HEPHero.cpp; then
  sed -i '/if( _SELECTION == "'${1}'" ) Setup'${1}'();/d' src/HEPHero.cpp
  sed -i '/if( _SELECTION == "'${1}'" && !'${1}'Region() ) Selected = false;/d' src/HEPHero.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) '${1}'Selection();/d' src/HEPHero.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) '${1}'Systematic();/d' src/HEPHero.cpp
  sed -i '/if( _SELECTION == "'${1}'" ) Finish'${1}'();/d' src/HEPHero.cpp
else
  echo "Setup${1} DOESN'T EXIST IN HEPHero.cpp. LEAVING FILE UNCHANGED"
fi


if grep -q "ana/${1}.cpp" CMakeLists.txt; then 
  sed -i "s~ ana/${1}.cpp~~" CMakeLists.txt
else
  echo "${1}.cpp DOESN'T EXIST IN CMakeLists.txt. LEAVING FILE UNCHANGED"
fi

#rm ana${1}.o
