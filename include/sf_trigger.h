/*
*
 * btageffanalyzer.h
 *
 * URL:      https://github.com/gabrielmscampos/btageffanalyzer
 * Version:  2.0.0
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


class sf_trigger {

    private:
        struct data {
            float pt_0_Min;
            float pt_0_Max;
            float pt_1_Min;
            float pt_1_Max;
            double SF;
            double SF_err_up;
            double SF_err_down;
            double SF_err_syst;
        };
        
        struct bounds {
            float pt_0_Min;
            float pt_0_Max;
            float pt_1_Min;
            float pt_1_Max;
        };

        double evalEfficiency(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            const auto &entries = dataMap.at(datasetName);
            for (const auto &e: entries) {
                // std::cout<<"Valor do subleading pt: "<<pt_1<<" valor do leading pt:  "<<pt_0<<std::endl;
                // std::cout<<"leading pt min: "<<e.pt_0_Min<<" leading pt max: "<<e.pt_0_Max<<" subleadin pt min: "<<e.pt_1_Min<<" subleading pt max: "<<e.pt_1_Max<<" valor eff: "<<e.SF<<std::endl;
                if (
                    pt_0 >= e.pt_0_Min &&
                    pt_0 <= e.pt_0_Max &&
                    pt_1 >= e.pt_1_Min &&
                    pt_1 < e.pt_1_Max
                ) {
                    return e.SF;
                }
            }
            return 0.;
        }

        double GetUpError(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            const auto &entries = dataMap.at(datasetName);
            for (const auto &e: entries) {
                // std::cout<<"Valor do subleading pt: "<<pt_1<<" valor do leading pt:  "<<pt_0<<std::endl;
                // std::cout<<"leading pt min: "<<e.pt_0_Min<<" leading pt max: "<<e.pt_0_Max<<" subleadin pt min: "<<e.pt_1_Min<<" subleading pt max: "<<e.pt_1_Max<<" valor eff: "<<e.SF<<std::endl;
                if (
                    pt_0 >= e.pt_0_Min &&
                    pt_0 <= e.pt_0_Max &&
                    pt_1 >= e.pt_1_Min &&
                    pt_1 < e.pt_1_Max
                ) {
                    return e.SF_err_up;
                }
            }
            return 0.;
        }
        double GetLowError(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            const auto &entries = dataMap.at(datasetName);
            for (const auto &e: entries) {
                // std::cout<<"Valor do subleading pt: "<<pt_1<<" valor do leading pt:  "<<pt_0<<std::endl;
                // std::cout<<"leading pt min: "<<e.pt_0_Min<<" leading pt max: "<<e.pt_0_Max<<" subleadin pt min: "<<e.pt_1_Min<<" subleading pt max: "<<e.pt_1_Max<<" valor eff: "<<e.SF<<std::endl;
                if (
                    pt_0 >= e.pt_0_Min &&
                    pt_0 <= e.pt_0_Max &&
                    pt_1 >= e.pt_1_Min &&
                    pt_1 < e.pt_1_Max
                ) {
                    return e.SF_err_down;
                }
            }
            return 0.;
        }

        double GetSystError(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            const auto &entries = dataMap.at(datasetName);
            for (const auto &e: entries) {
                // std::cout<<"Valor do subleading pt: "<<pt_1<<" valor do leading pt:  "<<pt_0<<std::endl;
                // std::cout<<"leading pt min: "<<e.pt_0_Min<<" leading pt max: "<<e.pt_0_Max<<" subleadin pt min: "<<e.pt_1_Min<<" subleading pt max: "<<e.pt_1_Max<<" valor eff: "<<e.SF<<std::endl;
                if (
                    pt_0 >= e.pt_0_Min &&
                    pt_0 <= e.pt_0_Max &&
                    pt_1 >= e.pt_1_Min &&
                    pt_1 < e.pt_1_Max
                ) {
                    return e.SF_err_syst;
                }
            }
            return 0.;
        }



    public:
        std::map<std::string, std::vector<data>> dataMap;
        std::map<std::string, bounds> boundsMap;

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
                float minLeadingBound = 0.; float maxLeadingBound = 0.;
                float minSubleadingBound = -1.; float maxSubleadingBound = -1.;
                for (int j = 0; j < dtValues.Size(); j++) {
                    data te;
                    te.pt_0_Max = dtValues[j]["lep0_pt_max"].GetFloat();
                    te.pt_0_Min = dtValues[j]["lep0_pt_min"].GetFloat();
                    te.pt_1_Min = dtValues[j]["lep1_pt_min"].GetFloat();
                    te.pt_1_Max = dtValues[j]["lep1_pt_max"].GetFloat();
                    te.SF = dtValues[j]["SF"].GetDouble();
                    te.SF_err_up = dtValues[j]["SF_err_up"].GetDouble();
                    te.SF_err_down = dtValues[j]["SF_err_down"].GetDouble();
                    te.SF_err_syst= dtValues[j]["SF_err_syst"].GetDouble();
                    dataMap[dt].push_back(te);
                    minLeadingBound = minLeadingBound < te.pt_0_Min ? minLeadingBound : te.pt_0_Min;
                    maxLeadingBound = maxLeadingBound > te.pt_0_Max ? maxLeadingBound : te.pt_0_Max;
                    minSubleadingBound = minSubleadingBound < te.pt_1_Min ? minSubleadingBound : te.pt_1_Min;
                    maxSubleadingBound = maxSubleadingBound > te.pt_1_Max ? maxSubleadingBound : te.pt_1_Max;
                    if (minSubleadingBound < 0.) {
                        minSubleadingBound = te.pt_1_Min;
                        maxSubleadingBound = te.pt_1_Max;
                    }
                }
                bounds tb;
                tb.pt_0_Min = minLeadingBound;
                tb.pt_0_Max = maxLeadingBound;
                tb.pt_1_Min = minSubleadingBound;
                tb.pt_1_Max = maxSubleadingBound;
                boundsMap[dt] = tb;
            }

            fclose(fp_cert);
        }

        double getSF(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            // Absolute value of leading lepton pt
            pt_0 = fabs(pt_0);
            // std::cout<<"leading pt: "<<pt_0<<std::endl;
            // std::cout<<"valor do bounds "<<boundsMap.at(datasetName).pt_0_Min<<"-"<<boundsMap.at(datasetName).pt_0_Max<<std::endl;
            // If pt is out of bounds, return 0
            if (
                pt_0 < boundsMap.at(datasetName).pt_0_Min
            ) return 0;

            if (
                pt_0 >= boundsMap.at(datasetName).pt_0_Max
            ){
                pt_0 = boundsMap.at(datasetName).pt_0_Max - .0001;
            }
            // If pt is out of bounds, define pt to evaluate next to bound
            float ptToEvaluate = pt_1;
            bool ptOutOfBounds = false;

            // If pt is lesser than minimum bound, return 0
            if (pt_1 < boundsMap.at(datasetName).pt_1_Min) return 0;


            // When given pT is greater then maximum boundary we compute
            // the efficiency at the maximum boundary
            if (pt_1 >= boundsMap.at(datasetName).pt_1_Max) {
                ptOutOfBounds = true;
                ptToEvaluate = boundsMap.at(datasetName).pt_1_Max - .0001;
            }

            // std::cout<<"subleading pt:  "<<pt_1<<std::endl;
            // std::cout<<"valor do bounds pt max "<<boundsMap.at(datasetName).pt_1_Max<<" e apos a correcao "<<ptToEvaluate<<std::endl;


            return evalEfficiency(datasetName, pt_0, ptToEvaluate);
        }
        double getSFErrorLow(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            // Absolute value of leading lepton pt
            pt_0 = fabs(pt_0);
            // std::cout<<"leading pt: "<<pt_0<<std::endl;
            // std::cout<<"valor do bounds "<<boundsMap.at(datasetName).pt_0_Min<<"-"<<boundsMap.at(datasetName).pt_0_Max<<std::endl;
            // If pt is out of bounds, return 0
            if (
                pt_0 < boundsMap.at(datasetName).pt_0_Min 
            ) return 0;

            if(
                pt_0 >= boundsMap.at(datasetName).pt_0_Max
            ){
                pt_0 = boundsMap.at(datasetName).pt_0_Max - .0001;
            }
            // If pt is out of bounds, define pt to evaluate next to bound
            float ptToEvaluate = pt_1;
            bool ptOutOfBounds = false;

            // If pt is lesser than minimum bound, return 0
            if (pt_1 < boundsMap.at(datasetName).pt_1_Min) return 0;


            // When given pT is greater then maximum boundary we compute
            // the efficiency at the maximum boundary
            if (pt_1 >= boundsMap.at(datasetName).pt_1_Max) {
                ptOutOfBounds = true;
                ptToEvaluate = boundsMap.at(datasetName).pt_1_Max - .0001;
            }

            // std::cout<<"subleading pt:  "<<pt_1<<std::endl;
            // std::cout<<"valor do bounds pt max "<<boundsMap.at(datasetName).pt_1_Max<<" e apos a correcao "<<ptToEvaluate<<std::endl;


            return GetLowError(datasetName, pt_0, ptToEvaluate);
        }
        double getSFErrorUp(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            // Absolute value of leading lepton pt
            pt_0 = fabs(pt_0);
            // std::cout<<"leading pt: "<<pt_0<<std::endl;
            // std::cout<<"valor do bounds "<<boundsMap.at(datasetName).pt_0_Min<<"-"<<boundsMap.at(datasetName).pt_0_Max<<std::endl;
            // If pt is out of bounds, return 0
            if (
                pt_0 < boundsMap.at(datasetName).pt_0_Min 
            ) return 0;

            if(
                pt_0 >= boundsMap.at(datasetName).pt_0_Max
            ){
                pt_0 = boundsMap.at(datasetName).pt_0_Max - .0001;
            }

            // If pt is out of bounds, define pt to evaluate next to bound
            float ptToEvaluate = pt_1;
            bool ptOutOfBounds = false;

            // If pt is lesser than minimum bound, return 0
            if (pt_1 < boundsMap.at(datasetName).pt_1_Min) return 0;


            // When given pT is greater then maximum boundary we compute
            // the efficiency at the maximum boundary
            if (pt_1 >= boundsMap.at(datasetName).pt_1_Max) {
                ptOutOfBounds = true;
                ptToEvaluate = boundsMap.at(datasetName).pt_1_Max - .0001;
            }

            // std::cout<<"subleading pt:  "<<pt_1<<std::endl;
            // std::cout<<"valor do bounds pt max "<<boundsMap.at(datasetName).pt_1_Max<<" e apos a correcao "<<ptToEvaluate<<std::endl;


            return GetUpError(datasetName, pt_0, ptToEvaluate);
        }
         double getSFErrorSys(
            std::string datasetName,
            float pt_0,
            float pt_1
        ) {
            // Absolute value of leading lepton pt
            pt_0 = fabs(pt_0);
            // std::cout<<"leading pt: "<<pt_0<<std::endl;
            // std::cout<<"valor do bounds "<<boundsMap.at(datasetName).pt_0_Min<<"-"<<boundsMap.at(datasetName).pt_0_Max<<std::endl;
            // If pt is out of bounds, return 0
            if (
                pt_0 < boundsMap.at(datasetName).pt_0_Min 
            ) return 0;

            if(
                pt_0 >= boundsMap.at(datasetName).pt_0_Max
            ){
                pt_0 = boundsMap.at(datasetName).pt_0_Max - .0001;
            }

            // If pt is out of bounds, define pt to evaluate next to bound
            float ptToEvaluate = pt_1;
            bool ptOutOfBounds = false;

            // If pt is lesser than minimum bound, return 0
            if (pt_1 < boundsMap.at(datasetName).pt_1_Min) return 0;


            // When given pT is greater then maximum boundary we compute
            // the efficiency at the maximum boundary
            if (pt_1 >= boundsMap.at(datasetName).pt_1_Max) {
                ptOutOfBounds = true;
                ptToEvaluate = boundsMap.at(datasetName).pt_1_Max - .0001;
            }

            // std::cout<<"subleading pt:  "<<pt_1<<std::endl;
            // std::cout<<"valor do bounds pt max "<<boundsMap.at(datasetName).pt_1_Max<<" e apos a correcao "<<ptToEvaluate<<std::endl;


            return GetSystError(datasetName, pt_0, ptToEvaluate);
        }
};