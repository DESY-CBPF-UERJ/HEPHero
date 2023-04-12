/*
 * btageffanalyzer.h
 *
 * URL:      https://github.com/gabrielmscampos/btageffanalyzer
 * Version:  4.0.0
 *
 * Copyright (C) 2021-2021 Gabriel Moreira da Silva Campos <gabrielmscampos@gmail.com>
 *
 * btageffanalyzer is distributed under the GPL-3.0 license, see LICENSE for details.
 *
 */

#include <iostream>
#include <vector>
#include <map>
#include <math.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

class BTagEffAnalyzer {

    private:
        struct data {
            float etaMin;
            float etaMax;
            float ptMin;
            float ptMax;
            double eff;
        };
        
        struct bounds {
            float etaMin;
            float etaMax;
            float ptMin;
            float ptMax;
        };

        double evalEfficiency(
            std::string hadronFlavour,
            float eta,
            float pt
        ) {
            const auto &entries = dataMap.at(datasetName).at(hadronFlavour);
            for (const auto &e: entries) {
                if (
                    eta >= e.etaMin &&
                    eta <= e.etaMax &&
                    pt >= e.ptMin &&
                    pt < e.ptMax
                ) {
                    return e.eff;
                }
            }
            return 0.;
        }

    public:
        std::map<std::string, std::map<std::string, std::vector<data>>> dataMap;
        std::map<std::string, std::map<std::string, bounds>> boundsMap;
        std::string datasetName;

        void readFile(
            std::string fpath
        ){
            rapidjson::Document effFileMap;
            FILE *fp_cert = fopen(fpath.c_str(), "r"); 
            char buf_cert[0XFFFF];
            rapidjson::FileReadStream input_cert(fp_cert, buf_cert, sizeof(buf_cert));
            effFileMap.ParseStream(input_cert);

            for (auto i = effFileMap.MemberBegin(); i != effFileMap.MemberEnd(); i++) {
                std::string dt = i->name.GetString();
                rapidjson::Value& dtValues = effFileMap[dt.c_str()];
                for (auto j = dtValues.MemberBegin(); j != dtValues.MemberEnd(); j++) {
                    std::string hadronFlavour = j->name.GetString();
                    rapidjson::Value& dtHfValues = dtValues[hadronFlavour.c_str()];
                    float minEtaBound = 0.; float maxEtaBound = 0.;
                    float minPtBound = -1.; float maxPtBound = -1.;
                    for (int k = 0; k < dtHfValues.Size(); k++) {
                        data te;
                        te.etaMin = dtHfValues[k]["eta_min"].GetFloat();
                        te.etaMax = dtHfValues[k]["eta_max"].GetFloat();
                        te.ptMin = dtHfValues[k]["pt_min"].GetFloat();
                        te.ptMax = dtHfValues[k]["pt_max"].GetFloat();
                        te.eff = dtHfValues[k]["eff"].GetDouble();
                        dataMap[dt][hadronFlavour].push_back(te);
                        minEtaBound = minEtaBound < te.etaMin ? minEtaBound : te.etaMin;
                        maxEtaBound = maxEtaBound > te.etaMax ? maxEtaBound : te.etaMax;
                        minPtBound = minPtBound < te.ptMin ? minPtBound : te.ptMin;
                        maxPtBound = maxPtBound > te.ptMax ? maxPtBound : te.ptMax;
                        if (minPtBound < 0.) {
                            minPtBound = te.ptMin;
                            maxPtBound = te.ptMax;
                        }
                    }
                    bounds tb;
                    tb.etaMin = minEtaBound;
                    tb.etaMax = maxEtaBound;
                    tb.ptMin = minPtBound;
                    tb.ptMax = maxPtBound;
                    boundsMap[dt][hadronFlavour] = tb;
                }
            }

            fclose(fp_cert);
        }

        void calib(
            std::string _datasetName,
            std::string _fallbackDataset
        ) {
            // Check if key `_datasetName` exists in the map
            if (dataMap.count(_datasetName) < 1) {
                std::cout << "WARNING: Missing dataset " + _datasetName + " in efficiency file. Fallbacking to " + _fallbackDataset + "." << std::endl;
                _datasetName = _fallbackDataset;
            }
            datasetName = _datasetName;
        }

        double getEfficiency(
            std::string hadronFlavour,
            float eta,
            float pt
        ) {
            // Absolute value of eta
            eta = fabs(eta);

            // If eta is out of bounds, return 0
            if (
                eta < boundsMap.at(datasetName).at(hadronFlavour).etaMin ||
                eta > boundsMap.at(datasetName).at(hadronFlavour).etaMax
            ) return 0;

            // If pt is out of bounds, define pt to evaluate next to bound
            float ptToEvaluate = pt;
            bool ptOutOfBounds = false;

            // If pt is lesser than minimum bound, return 0
            if (pt < boundsMap.at(datasetName).at(hadronFlavour).ptMin) return 0;

            // When given pT is greater then maximum boundary we compute
            // the efficiency at the maximum boundary
            if (pt >= boundsMap.at(datasetName).at(hadronFlavour).ptMax) {
                ptOutOfBounds = true;
                ptToEvaluate = boundsMap.at(datasetName).at(hadronFlavour).ptMax - .0001;
            }

            return evalEfficiency(hadronFlavour, eta, ptToEvaluate);
        }
};
