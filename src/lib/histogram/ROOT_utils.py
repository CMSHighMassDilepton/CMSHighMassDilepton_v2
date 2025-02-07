import ROOT
import numpy as np

def setTDRStyle():
  ROOT.gStyle.SetCanvasBorderMode(0);
  ROOT.gStyle.SetCanvasColor(0);
  ROOT.gStyle.SetCanvasDefH(600);
  ROOT.gStyle.SetCanvasDefW(600);
  ROOT.gStyle.SetCanvasDefX(0);
  ROOT.gStyle.SetCanvasDefY(0);

  ROOT.gStyle.SetPadBorderMode(0);
  ROOT.gStyle.SetPadColor(0); 
  ROOT.gStyle.SetPadGridX(0);
  ROOT.gStyle.SetPadGridY(0);
  ROOT.gStyle.SetGridColor(0);
  ROOT.gStyle.SetGridStyle(3);
  ROOT.gStyle.SetGridWidth(1);

  ROOT.gStyle.SetFrameBorderMode(0);
  ROOT.gStyle.SetFrameBorderSize(1);
  ROOT.gStyle.SetFrameFillColor(0);
  ROOT.gStyle.SetFrameFillStyle(0);
  ROOT.gStyle.SetFrameLineColor(1);
  ROOT.gStyle.SetFrameLineStyle(1);
  ROOT.gStyle.SetFrameLineWidth(1);
  ROOT.gStyle.SetHistLineColor(1);
  ROOT.gStyle.SetHistLineStyle(0);
  ROOT.gStyle.SetHistLineWidth(1);

  ROOT.gStyle.SetEndErrorSize(2);
  ROOT.gStyle.SetFuncColor(2);
  ROOT.gStyle.SetFuncStyle(1);
  ROOT.gStyle.SetFuncWidth(1);
  ROOT.gStyle.SetOptDate(0);
  
  ROOT.gStyle.SetOptFile(0);
  ROOT.gStyle.SetOptStat(0);
  ROOT.gStyle.SetStatColor(0); 
  ROOT.gStyle.SetStatFont(42);
  ROOT.gStyle.SetStatFontSize(0.04);
  ROOT.gStyle.SetStatTextColor(1);
  ROOT.gStyle.SetStatFormat("6.4g");
  ROOT.gStyle.SetStatBorderSize(1);
  ROOT.gStyle.SetStatH(0.1);
  ROOT.gStyle.SetStatW(0.15);

  ROOT.gStyle.SetPadTopMargin(0.07);
  ROOT.gStyle.SetPadBottomMargin(0.13);
  ROOT.gStyle.SetPadLeftMargin(0.12);
  ROOT.gStyle.SetPadRightMargin(0.05);

  ROOT.gStyle.SetOptTitle(0);
  ROOT.gStyle.SetTitleFont(42);
  ROOT.gStyle.SetTitleColor(1);
  ROOT.gStyle.SetTitleTextColor(1);
  ROOT.gStyle.SetTitleFillColor(10);
  ROOT.gStyle.SetTitleFontSize(0.05);

  ROOT.gStyle.SetTitleColor(1, "XYZ");
  ROOT.gStyle.SetTitleFont(42, "XYZ");
  ROOT.gStyle.SetTitleSize(0.05, "XYZ");
  ROOT.gStyle.SetTitleXOffset(0.9);
  ROOT.gStyle.SetTitleYOffset(1.05);
 
  ROOT.gStyle.SetLabelColor(1, "XYZ");
  ROOT.gStyle.SetLabelFont(42, "XYZ");
  ROOT.gStyle.SetLabelOffset(0.007, "XYZ");
  ROOT.gStyle.SetLabelSize(0.04, "XYZ");

  ROOT.gStyle.SetAxisColor(1, "XYZ");
  ROOT.gStyle.SetStripDecimals(1); 
  ROOT.gStyle.SetTickLength(0.025, "XYZ");
  ROOT.gStyle.SetNdivisions(510, "XYZ");
  ROOT.gStyle.SetPadTickX(1); 
  ROOT.gStyle.SetPadTickY(1);

  ROOT.gStyle.SetOptLogx(0);
  ROOT.gStyle.SetOptLogy(0);
  ROOT.gStyle.SetOptLogz(0);

  ROOT.gStyle.SetPaperSize(20.,20.);
  ROOT.gStyle.SetPaintTextFormat(".2f");

def CMS_lumi( pad,  lumi,  up = False,  status = "", reduceSize = False, offset = 0,offsetLumi = 0):
  latex2 = ROOT.TLatex();
  latex2.SetNDC();
  latex2.SetTextSize(0.6*pad.GetTopMargin());
  latex2.SetTextFont(42);
  latex2.SetTextAlign(31);
  if(reduceSize):
    latex2.SetTextSize(0.5*pad.GetTopMargin());
  
  if(lumi != ""):
    latex2.DrawLatex(0.94+offsetLumi, 0.95,(lumi+" fb^{-1} (13 TeV)"));
  else:
    latex2.DrawLatex(0.88+offsetLumi, 0.95,(lumi+"(13 TeV)"));

  if(up):
    latex2.SetTextSize(0.65*pad.GetTopMargin());
    if(reduceSize):
      latex2.SetTextSize(0.5*pad.GetTopMargin());
    latex2.SetTextFont(62);
    latex2.SetTextAlign(11);    
    latex2.DrawLatex(0.15+offset, 0.95, "CMS");
  else:
    latex2.SetTextSize(0.6*pad.GetTopMargin());
    if(reduceSize):
      latex2.SetTextSize(0.45*pad.GetTopMargin());
    elif(reduceSize == 2):
      latex2.SetTextSize(0.40*pad.GetTopMargin());

    latex2.SetTextFont(62);
    latex2.SetTextAlign(11);    
    latex2.DrawLatex(0.175+offset, 0.86, "CMS");

  if(status != ""):
    
    if(up):
      latex2.SetTextSize(0.55*pad.GetTopMargin());
      latex2.SetTextFont(52);
      latex2.SetTextAlign(11);
      latex2.DrawLatex(0.235+offset, 0.95, status);
    
    else:
      latex2.SetTextSize(0.6*pad.GetTopMargin());
      if(reduceSize):
          latex2.SetTextSize(0.45*pad.GetTopMargin());
      latex2.SetTextFont(52);
      latex2.SetTextAlign(11);    
      if(reduceSize):
          latex2.DrawLatex(0.235+offset, 0.86, status);
      else:
          latex2.DrawLatex(0.28+offset, 0.86, status);

# def reweightROOTH(hist, weight: float):
#     """
#     reweight the histogram values and its errors
#     the given weight value
#     """
#     for idx in range(1, hist.GetNbinsX()+1):
#         hist.SetBinContent(idx, hist.GetBinContent(idx)*weight)
#         hist.SetBinError(idx, hist.GetBinError(idx)*weight)
#     return

def reweightROOTH_data(hist, weight: float):
    """
    reweight the histogram values keep its original relative
    error by multiplying its errors to the appropriate values
    """
    for idx in range(1, hist.GetNbinsX()+1):
        val_orig = hist.GetBinContent(idx)
        err_orig = hist.GetBinError(idx)
        if val_orig != 0:
            rel_err = err_orig/val_orig
        else:
            rel_err = 0
        val_new = val_orig*weight
        err_new = val_new*rel_err
        hist.SetBinContent(idx, val_new)
        hist.SetBinError(idx, err_new)
    return

def reweightROOTH_mc(hist, weight: float):
    """
    reweight histogram errors according to the fraction
    weight. This is different from data since MC samples
    are normalized by cross section * lumi, regardless of the sample
    size. Therefore, the error isn't properly propagated
    """
    for idx in range(1, hist.GetNbinsX()+1):
        val_new = hist.GetBinContent(idx) # current value is already new value
        val_orig = val_new/weight
        err_orig = np.sqrt(val_orig)
        if val_orig != 0:
            rel_err = err_orig/val_orig
        else:
            rel_err = 0 

        # print(f"original val at idx {idx}: {val_new}")
        if val_new !=0:
            print(f"original rel err idx {idx}: {rel_err}")
            print(f"new auto rel err idx {idx}: {hist.GetBinError(idx)/val_new}")
        err_new = val_new*rel_err
        # if val_new !=0:
        #     print(f"err_new idx {idx}: {err_new}")
        #     print(f"original bad err idx {idx}: {hist.GetBinError(idx)}")
        hist.SetBinError(idx, err_new)
    return