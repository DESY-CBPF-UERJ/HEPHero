// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>> Helper for ttbar kinematic reconstruction (kinreco) >>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// The basic problem is to recover top and antitiop momenta in the 
// dilepton decay, having measured two leptons, jets (possibly more 
// than 2) and missing transverse energy, see description-ttbar.pdf, 
// page 4. Follows the method described in Phys. Rev. D73 (2006) 054015,
// additional information can be found in DESY-THESIS-2012-037

#ifndef TTBAR_KINRECO_H
#define TTBAR_KINRECO_H

// C++ library or ROOT header files
#include <TMath.h>
#include <Math/Polynomial.h>
#include <TLorentzVector.h>
#include <TH1D.h>
#include <vector>
#include <complex>

// debugging level (0 for silence, > 0 for some messages)
int gDebug = 0;

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>> ZSolutionKinRecoDilepton >>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// structure to store one solution of the kinematic reconstruction
struct ZSolutionKinRecoDilepton
{
  // constructor
  // (set weight to -1 by default)
  ZSolutionKinRecoDilepton(): zWeight(-1.0) {;}
  // top and antitop four momenta
  TLorentzVector zT, zTbar;
  // mumber of b-tagged jets (can be 0, 1 or 2)
  int zBTag;
  // weight of this solution
  // (calculated according to neutrino momentum spectrum, see below)
  double zWeight;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>> SolveKinRecoDilepton routine >>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// Routine to solve the kinreco problem for given b, bbar jets
// Arguments:
//    const TLorentzVector& lm:   lepton- momentum
//    const TLorentzVector& lp:   lepton+ momentum
//    const TLorentzVector& b:    b momentum
//    const TLorentzVector& bbar: bbar momentum
//    const double metX:          x-component of missing transverse energy (MET)
//    const double metY:          y-component of missing transverse energy
//    TH1D* hInacc = NULL:        histogram to be filled with the calculated inaccuracy (for debugging purpose, not filled by default)
//    TH1D* ambiguity = NULL:     histogram to be filled with the number of ambiguities (for debugging purpose, not filled by default)
// Returns the ZSolutionKinRecoDilepton pointer (see above)
// For math, see Lars Sonnenschein's paper Phys.Rev. D73 (2006) 054015 [Erratum Phys.Rev. D73 (2006) 054015]
ZSolutionKinRecoDilepton* SolveKinRecoDilepton(const TLorentzVector& lm, const TLorentzVector& lp, 
  const TLorentzVector& b, const TLorentzVector& bbar, const double metX, const double metY, 
  TH1D* hInacc = NULL, int* ambiguity = NULL)
{
  // constants
  const double massW = 80.4; // W boson mass
  const double massTop = 172.5; // top quark mass
  double landauMean = 58.0; // mean of Landau distribution for neutrino momentum spectrum (see DESY-THESIS-2012-037)
  double landauSigma = 22.0; // sigma of Landau distribution for neutrino momentum spectrum (see DESY-THESIS-2012-037)
  double epsForCheck = 1e+0; // threshold for numerical precison checks (for debugging purpose)
  
  // Transform input into double variables with short names
  // jet1 (b)
  double xb = b.X();
  double yb = b.Y();
  double zb = b.Z();
  double mb = b.M();
  double mb2 = mb * mb;
  double eb = b.E();
  double eb2 = eb * eb;
  // jet2 (bbar)
  double xbbar = bbar.X();
  double ybbar = bbar.Y();
  double zbbar = bbar.Z();
  double mbbar = bbar.M();
  double mbbar2 = mbbar * mbbar;
  double ebbar = bbar.E();
  double ebbar2 = ebbar * ebbar;
  // el (lm)
  double xlm = lm.X();
  double xlm2 = xlm * xlm;
  double ylm = lm.Y();
  double ylm2 = ylm * ylm;
  double zlm = lm.Z();
  double zlm2 = zlm * zlm;
  double mlm = lm.M();
  double mlm2 = mlm * mlm;
  double elm = lm.E();
  double elm2 = elm * elm;
  // mu (lp)
  double xlp = lp.X();
  double xlp2 = xlp * xlp;
  double ylp = lp.Y();
  double ylp2 = ylp * ylp;
  double zlp = lp.Z();
  double zlp2 = zlp * zlp;
  double mlp = lp.M();
  double mlp2 = mlp * mlp;
  double elp = lp.E();
  double elp2 = elp * elp;
  // MET
  double ex = metX;
  double ex2 = ex * ex;
  double ey = metY;
  double ey2 = ey * ey;
  // constraints
  double mw2 = massW * massW;
  double mt2 = massTop * massTop;
  double mn2 = 0.0;
  
  // Calculate coefficients from Lars' paper
  // a coefs
  double a1 = (eb + elp) * (mw2 - mlp2 - mn2) - elp * (mt2 - mb2 - mlp2 - mn2) + 2 * eb * elp2 - 2 * elp * (xb * xlp + yb * ylp + zb * zlp);
  double a12 = a1 * a1;
  double a2 = 2 * (eb * xlp - elp * xb);
  double a22 = a2 * a2;
  double a3 = 2 * (eb * ylp - elp * yb);
  double a32 = a3 * a3;
  double a4 = 2 * (eb * zlp - elp * zb);
  double a42 = a4 * a4;
  // b coefs
  double b1 = (ebbar + elm) * (mw2 - mlm2 - mn2) - elm * (mt2 - mbbar2 - mlm2 - mn2) + 2 * ebbar * elm2 - 2 * elm * (xbbar * xlm + ybbar * ylm + zbbar * zlm);
  double b12 = b1 * b1;
  double b2 = 2 * (ebbar * xlm - elm * xbbar);
  double b22 = b2 * b2;
  double b3 = 2 * (ebbar * ylm - elm * ybbar);
  double b32 = b3 * b3;
  double b4 = 2 * (ebbar * zlm - elm * zbbar);
  double b42 = b4 * b4;
  // c coefs
  double c22 = TMath::Power(mw2 - mlp2 - mn2, 2.0) - 4 * (elp2 - zlp2) * a12 / a42 - 4 * (mw2 - mlp2 - mn2) * zlp * a1 / a4;
  double c21 = 4 * (mw2 - mlp2 - mn2) * (xlp - zlp * a2 / a4) - 8 * (elp2 - zlp2) * a1 * a2 / a42 - 8 * xlp * zlp * a1 / a4;
  double c20 = -4 * (elp2 - xlp2) - 4 * (elp2 - zlp2) * a22 / a42 - 8 * xlp * zlp * a2 / a4;
  double c11 = 4 * (mw2 - mlp2 - mn2) * (ylp - zlp * a3 / a4) - 8 * (elp2 - zlp2) * a1 * a3 / a42 - 8 * ylp * zlp * a1 / a4;
  double c10 = -8 * (elp2 - zlp2) * a2 * a3 / a42 + 8 * xlp * ylp - 8 * xlp * zlp * a3 / a4 - 8 * ylp * zlp * a2 / a4;
  double c00 = -4 * (elp2 - ylp2) - 4 * (elp2 - zlp2) * a32 / a42 - 8 * ylp * zlp * a3 / a4;
  // d' coefs
  double d22p = TMath::Power(mw2 - mlm2 - mn2, 2.0) - 4 * (elm2 - zlm2) * b12 / b42 - 4 * (mw2 - mlm2 - mn2) * zlm * b1 / b4;
  double d21p = 4 * (mw2 - mlm2 - mn2) * (xlm - zlm * b2 / b4) - 8 * (elm2 - zlm2) * b1 * b2 / b42 - 8 * xlm * zlm * b1 / b4;
  double d20p = -4 * (elm2 - xlm2) - 4 * (elm2 - zlm2) * b22 / b42 - 8 * xlm* zlm * b2 / b4;
  double d11p = 4 * (mw2 - mlm2 - mn2) * (ylm - zlm * b3 / b4) - 8 * (elm2 - zlm2) * b1 * b3 / b42 - 8 * ylm * zlm * b1 / b4;
  double d10p = -8 * (elm2 - zlm2) * b2 * b3 / b42 + 8 * xlm * ylm - 8 * xlm * zlm * b3 / b4 - 8 * ylm * zlm * b2 / b4;
  double d00p = -4 * (elm2 - ylm2) - 4 * (elm2 - zlm2) * b32 / b42 - 8 * ylm * zlm * b3 / b4;
  // d coefs
  double d22 = d22p + ex2 * d20p + ey2 * d00p + ex * ey * d10p + ex * d21p + ey * d11p;
  double d21 = - d21p - 2 * ex * d20p - ey * d10p;
  double d20 = d20p;
  double d11 = - d11p - 2 * ey * d00p - ex * d10p;
  double d10 = d10p;
  double d00 = d00p;
  // h coefs
  double h4 = c00 * c00 * d22 * d22 + c11 * d22 * (c11 * d00 - c00 * d11) 
            + c00 * c22 * (d11 * d11 - 2 * d00 * d22) + c22 * d00 * (c22 * d00 - c11 * d11);
  double h3 = c00 * d21 * (2 * c00 * d22 - c11 * d11) + c00 * d11 * (2 * c22 * d10 + c21 * d11)
            + c22 * d00 * (2 * c21 * d00 - c11 * d10) - c00 * d22 * (c11 * d10 + c10 * d11) 
            -2 * c00 * d00 * (c22 * d21 + c21 * d22) - d00 * d11 * (c11 * c21 + c10 * c22) 
            + c11 * d00 * (c11 * d21 + 2 * c10 * d22);
  double h2 = c00 * c00 * (2 * d22 * d20 + d21 * d21) - c00 * d21 * (c11 * d10 + c10 * d11)
            + c11 * d20 * (c11 * d00 - c00 * d11) + c00 * d10 * (c22 * d10 - c10 * d22)
            + c00 * d11 * (2 * c21 * d10 + c20 * d11) + (2 * c22 * c20 + c21 * c21) * d00 * d00
            - 2 * c00 * d00 * (c22 * d20 + c21 * d21 + c20 * d22)
            + c10 * d00 * (2 * c11 * d21 + c10 * d22) - d00 * d10 * (c11 * c21 + c10 * c22)
            - d00 * d11 * (c11 * c20 + c10 * c21);
  double h1 = c00 * d21 * (2 * c00 * d20 - c10 * d10) - c00 * d20 * (c11 * d10 + c10 * d11)
            + c00 * d10 * (c21 * d10 + 2 * c20 * d11) - 2 * c00 * d00 * (c21 * d20 + c20 * d21)
            + c10 * d00 * (2 * c11 * d20 + c10 * d21) + c20 * d00 * (2 * c21 * d00 - c10 * d11) // this is correct
            //+ c10 * d00 * (2 * c11 * d20 + c10 * d21) - c20 * d00 * (2 * c21 * d00 - c10 * d11) // this is wrong
            - d00 * d10 * (c11 * c20 + c10 * c21);
  double h0 = c00 * c00 * d20 * d20 + c10 * d20 * (c10 * d00 - c00 * d10)
            + c20 * d10 * (c00 * d10 - c10 * d00) + c20 * d00 * (c20 * d00 - 2 * c00 * d20);
  
  // solve quartic equation
  ROOT::Math::Polynomial eq(4);
  double pars[5] = { h4, h3, h2, h1, h0 };
  // apply globale scaling to avoid possible numerical precision problems
  double minpar = 1e100;
  for(int p = 0; p < 5; p++)
    if(TMath::Abs(pars[p]) < minpar)
      minpar = pars[p];
  for(int p = 0; p < 5; p++)
    pars[p] /= minpar;
  //printf("parameters: %e %e %e %e %e\n", pars[0], pars[1], pars[2], pars[3], pars[4]);
  eq.SetParameters(pars);
  std::vector<double> roots = eq.FindRealRoots();
  if(gDebug)
    printf("N roots: %ld\n", roots.size());
  
  // restore all nu and nubar momenta components (see again Lars' paper)
  TLorentzVector nu, nubar, nuBest, nubarBest;
  double weightBest = -1.0;
  for(int s = 0; s < roots.size(); s++)
  {
    // check main equation
    double sol = pars[0] + pars[1]*roots[s] + pars[2]*roots[s]*roots[s] + pars[3]*roots[s]*roots[s]*roots[s] + pars[4]*roots[s]*roots[s]*roots[s]*roots[s];
    //printf("x: %e  solution: %e\n", roots[s], sol);

    // x components
    double xn = roots[s];
    double xnbar = ex - xn;
    // y components
    double c0 = c00;
    double c1 = c11 + c10 * xn;
    double c2 = c22 + c21 * xn + c20 * xn * xn;
    double d0 = d00;
    double d1 = d11 + d10 * xn;
    double d2 = d22 + d21 * xn + d20 * xn * xn;
    double yn = (c0 * d2 - c2 * d0) / (c1 * d0 - c0 * d1);
    double ynbar = ey - yn;
    // z components
    double zn = - (a1 + a2 * xn + a3 * yn) / a4;
    double znbar = - (b1 + b2 * xnbar + b3 * ynbar) / b4;
    // check nan
    if(xn != xn || xnbar != xnbar || yn != yn || ynbar != ynbar || zn != zn || znbar != znbar)
    {
      printf("SolveKinRecoDilepton: nan solution\n");
      //exit(1);
      continue;
    }
    TLorentzVector nu, nubar;
    nu.SetXYZM(xn, yn, zn, 0.0);
    nubar.SetXYZM(xnbar, ynbar, znbar, 0.0);
    // check solution
    TLorentzVector wp = lp + nu;
    TLorentzVector wm = lm + nubar;
    TLorentzVector t = wp + b;
    TLorentzVector tbar = wm + bbar;
    if(gDebug)
      printf("%e %e %e %e %e %e\n", wp.M(), wm.M(), t.M(), tbar.M(), nu.X() + nubar.X() - ex, nu.Y() + nubar.Y() - ey);
    // below are some calculations done for debugging purpose
    double inaccuracy = TMath::Abs(wp.M() - massW) + TMath::Abs(wm.M() - massW)
                      + TMath::Abs(t.M() - massTop) + TMath::Abs(tbar.M() - massTop) 
                      + TMath::Abs((nu + nubar).X() - ex) + TMath::Abs((nu + nubar).Y() - ey);
    if(hInacc)
      hInacc->Fill(inaccuracy);
    if(inaccuracy > epsForCheck )
    {
      //printf("%e %e %e %e %e %e\n", wp.M(), wm.M(), t.M(), tbar.M(), nu.X() + nubar.X() - ex, nu.Y() + nubar.Y() - ey);
      //printf("SolveKinRecoDilepton: inaccuracy %f > %e\n", inaccuracy, epsForCheck);
      //exit(1);
      //continue;
    }
    else 
    {
      if(gDebug)
        printf("inaccuracy: %f\n", inaccuracy);
    }
    // calculate weight according to nu and nubar momenta (see DESY-THESIS-2012-037)
    double wnu = TMath::Landau(nu.E(), landauMean, landauSigma);
    double wnubar = TMath::Landau(nubar.E(), landauMean, landauSigma);
    double weight = wnu * wnubar;
    if(gDebug)
      printf("nu e: %f %f  weight: %f\n", nu.E(), nubar.E(), weight);
    // update solution, if this is the best weight
    if(weight > weightBest)
    {
      weightBest = weight;
      nuBest = nu;
      nubarBest = nubar;
    }
    // for debugging purpose, if needed
    if(ambiguity)
      (*ambiguity)++;
  }
  
  // if the best weight is default negative, there is no solution
  if(weightBest < 0.0)
    return NULL;
  
  // store and return best solution as ZSolutionKinRecoDilepton instance
  ZSolutionKinRecoDilepton* solution = new ZSolutionKinRecoDilepton;
  solution->zT = (nuBest + lp + b);
  solution->zTbar = (nubarBest + lm + bbar);
  solution->zWeight = weightBest;
  return solution;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>> KinRecoDilepton routine >>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// Routine to solve the kinreco problem for the whole event (possibly with more than 2 jets)
// Arguments:
//    const TLorentzVector& lm:   lepton- momentum
//    const TLorentzVector& lp:   lepton+ momentum
//    const std::vector<TLorentzVector>& jets: container with jet momenta
//    const double metX:          x-component of missing transverse energy (MET)
//    const double metY:          y-component of missing transverse energy
//    const TLorentzVector& t:    top momentum (output)
//    const TLorentzVector& tbar: top momentum (output)
//    TH1D* hInacc = NULL:        histogram to be filled with the calculated inaccuracy (for debugging purpose, not filled by default, see there usage in SolveKinRecoDilepton())
//    TH1D* ambiguity = NULL:     histogram to be filled with the number of ambiguities (for debugging purpose, not filled by default, see there usage in SolveKinRecoDilepton())
// Returns 1 for successfull kinreco, 0 otherwise
// 
int KinRecoDilepton(const TLorentzVector& lm, const TLorentzVector& mp, const std::vector<TLorentzVector>& jets, 
  const double metX, const double metY, TLorentzVector& t, TLorentzVector& tbar, TH1D* hInacc = NULL, TH1D* hAmbig = NULL)
{
  // solution status (to be returned)
  int solved = 0;
  // best number of b-tagged jets (maximum 2)
  int bTagBest = 0;
  // best (largest) solution weight
  double weightBest = 0.0;
  
  // container with solutions
  std::vector<ZSolutionKinRecoDilepton*> vSolutions;
  // ambiguity and hAmbig pointers are for debugging purpose, not used normally
  int* ambiguity = NULL;
  if(hAmbig)
    ambiguity = new int(0);
    
  // print the number of jets if needed
  if(gDebug)
    printf("N jets: %ld\n", jets.size());
  
  // loop over 1st jet
  for(std::vector<TLorentzVector>::const_iterator jet1 = jets.begin(); jet1 != jets.end(); jet1++)
  {
    // skip if already have solution(s) with two b-tagged jets
    //if(jet1->pt() > 0 && bTagBest > 0)
    //  continue;
    // loop over 2nd jet
    for(std::vector<TLorentzVector>::const_iterator jet2 = jets.begin(); jet2 != jets.end(); jet2++)
    {
      // skip same jets
      if(jet1 == jet2) continue;
      // for this pair of jets, calculate number of b-tagged jets,
      // b-tagged jets are provided with negative masses (see selection.h):
      // account for this, then switch their masses to normal
      int bTagThis = 0;
      TLorentzVector jetB, jetBbar;
      if(jet1->M() < 0)
      {
        TLorentzVector jet;
        jet.SetPtEtaPhiM(jet1->Pt(), jet1->Eta(), jet1->Phi(), -1 * jet1->M());
        jetB = jet;
        bTagThis++;
      }
      else
        jetB = *jet1;
      if(jet2->M() < 0)
      {
        TLorentzVector jet;
        jet.SetPtEtaPhiM(jet2->Pt(), jet2->Eta(), jet2->Phi(), -1 * jet2->M());
        jetBbar = jet;
        bTagThis++;
      }
      else
        jetBbar = *jet2;
      // get solution
      TLorentzVector tThis, tbarThis;
      ZSolutionKinRecoDilepton* solution = SolveKinRecoDilepton(lm, mp, jetB, jetBbar, metX, metY, hInacc, ambiguity);
      if(!solution || solution->zWeight < 0)
        continue;
      // set b-tagging number
      solution->zBTag = bTagThis;
      // push to the container
      vSolutions.push_back(solution);
    }
  }
  
  // find best solution, preference order:
  //   with 2 b-tagged jets, if no then
  //   with 1 b-tagged jet, if no then
  //   with 0 b-tagged jets.
  // If more than one solution with the same number of b-tagged jets 
  // is available, take the solution with the largest weight 
  // (calculated according to the neutrino momenta spectrum, 
  // see DESY-THESIS-20120-037)
  for(std::vector<ZSolutionKinRecoDilepton*>::iterator it = vSolutions.begin(); it != vSolutions.end(); it++)
  {
    ZSolutionKinRecoDilepton* sol = *it;
    // worse b-tagging
    if(sol->zBTag < bTagBest)
      continue;
    // better b-tagging
    else if(sol->zBTag > bTagBest)
    {
      bTagBest = sol->zBTag;
      t = sol->zT;
      tbar = sol->zTbar;
      solved = 1;
    }
    // same b-tagging: check weight
    else
    {
      if(sol->zWeight > weightBest)
      {
        weightBest = sol->zWeight;
        t = sol->zT;
        tbar = sol->zTbar;
        solved = 1;
      }
    }
  }
  if(solved && hAmbig)
  {
    // for debugging purpose, if needed
    hAmbig->Fill(*ambiguity);
    delete ambiguity;
  }
  
  // all done, clear memory and return
  for(std::vector<ZSolutionKinRecoDilepton*>::iterator it = vSolutions.begin(); it != vSolutions.end(); it++)
    delete (*it);
  return solved;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


// routines below are not used in the analysis, 
// available for debugging purpose

// to generate decay a -> b, c
void Decay(const TLorentzVector& a, const double mb, const double mc, const double r1, const double r2, 
           TLorentzVector& b, TLorentzVector& c)
{
  double ma = a.M();
  double ma2 = ma * ma;
  double mb2 = mb * mb;
  double mc2 = mc * mc;
  double num = TMath::Power(ma2 - mb2 - mc2, 2.0) - 4 * mb2 * mc2;
  double denom = 4 * ma2;
  double p = TMath::Sqrt(num / denom);
  printf("p = %f\n", p);
  printf("sqrt(p2 + mb2) + sqrt(p2 + mc2) = a.E(): %f + %f = %f\n", TMath::Sqrt(p * p + mb2), TMath::Sqrt(p * p + mc2), a.M());
  double theta = TMath::Pi() * r1;
  double phi = 2 * TMath::Pi() * r2;
  b.SetXYZM(p * TMath::Cos(theta) * TMath::Cos(phi), p * TMath::Cos(theta) * TMath::Sin(phi), p * TMath::Sin(theta), mb);
  c.SetXYZM(- p * TMath::Cos(theta) * TMath::Cos(phi), - p * TMath::Cos(theta) * TMath::Sin(phi), - p * TMath::Sin(theta), mc);
  //b = b + a;
  //c = c + a;
  TVector3 boost = a.BoostVector();
  b.Boost(boost);
  c.Boost(boost);
}

// print four vector contents
void PrintTLV(const TString& str, const TLorentzVector& t)
{
  printf("%10s%10.3f%10.3f%10.3f%10.3f\n", str.Data(), t.X(), t.Y(), t.Z(), t.M());
}

#endif
