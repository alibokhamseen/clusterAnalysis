#Finds Distance estimates for a specified cluster
#Error propigation via taylor approximation (assuming symettric gaussian errors)
    #TODO: Change to finer Spectral class bins
    #TODO: get virial mass
    #TODO: add cepheid PL(C) distance estimate

    #TODO: get better metallicity source
    #TODO: add pipeline for queries (gaia, simbad)
    #TODO: Use cubic formula for photometric system conversions (or otherwise improve the algorithm)

    #TODO: add ZAMS MS fit
    #TODO: add isochrone fit, also get age
    #TODO: add better cluster membership filtering with algorithms


#Imports
from ssl import PEM_cert_to_DER_cert
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as stats_sci
import statistics as stat
import sys
from uncertainties import ufloat
from uncertainties.umath import *

#Stopping printing of warnings
pd.set_option("mode.chained_assignment", None)

#Main
def main():
#----------SET USER PARAMETERS HERE---------
    #Set z-score for pm exclusion
    z=1
    #Choose GAIA CSV input file name
    GAIAname = "Hyades-result.csv"
    SIMBADname = "simbad.txt"
#----------SET USER PARAMETERS HERE---------


    #Import Query Results, see GAIA and SIMBAD documentation for csv file variable names
    gaia = pd.read_csv(GAIAname)
    simbad = pd.read_csv(SIMBADname,sep="|")
    simbad = simbad.rename(columns=lambda x: x.strip())
    simbad = simbad.rename(columns=lambda x: "spec_type" if x=="spec. type" else x)

    #Parse Simbad data
    simbad["spec_type"] = simbad["spec_type"].str.strip()
        #Keep only main sequence stars (luminosity class 5) with spectral types
    simbad = simbad[simbad["spec_type"].str.match("^[OBAFGKM]\d+(\.\d*)?V$",na=False)]
        #remove duplicates by Ra, Dec
    simbad.drop_duplicates(subset=["coord1 (ICRS,J2000/2000)"])
        #Ignoring 'A value is trying to be set on a copy of a slice from a DataFrame.' warnings, this is what I am trying to do
        #Conver Ra,Dec to decimal degrees
    simbad["ra"] = simbad["coord1 (ICRS,J2000/2000)"].map(lambda x: float(x.split()[0])*(15) + float(x.split()[1])*(15/60) + float(x.split()[2])*(15/3600))
    simbad["Dec"] = simbad["coord1 (ICRS,J2000/2000)"].map(lambda x: float(x.split()[3]) + float(x.split()[4])*(1/60) + float(x.split()[5])*(1/3600))
        #Drop old coordinate format
    simbad = simbad.drop(columns=["coord1 (ICRS,J2000/2000)"])

    #Filtering gaia data for nonvalues and similar apparant magnitudes
    maxMag = 10
    gaia = gaia.dropna(subset=["phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag","parallax"])
    gaia = gaia[(gaia["phot_g_mean_mag"] < maxMag) & (gaia["phot_bp_mean_mag"] < maxMag) & (gaia["phot_rp_mean_mag"] < maxMag) & (gaia["parallax"] > 0)]

    #Get trig parallax distance to each star
    gaia["dist_trig_parallax"] = 1/(gaia["parallax"]/1000)
    gaia["err_dist_trig_parallax"] = np.abs(-1000/(gaia["parallax"]**2)*gaia["parallax_error"])

    #print different databases' ra, dec
    plt.figure()
    plt.title("Ra, Dec of Database Stars")
    plt.scatter(gaia["ra"],gaia["dec"],label="gaia",color='blue')
    plt.scatter(simbad["ra"],simbad["Dec"],label='simbad',color='red')
    plt.xlabel("Ra [degrees]")
    plt.ylabel("Dec [degrees]")
    plt.legend()
    plt.show()

    #Join gaia and simbad data
    data = coordMatch(gaia,simbad)
        #Delete old dataframes
    del gaia
    del simbad
    
    #Plot proper motions with z std circled
    plt.figure()
    plt.title("PMs of Cluster and Field")
    plt.xlabel("PM RA [units?]")
    plt.ylabel("PM Dec [units?]")
    plt.scatter(data["pmra"],data["pmdec"])
    ra_mean = stat.mean(data["pmra"])
    dec_mean = stat.mean(data["pmdec"])
    ra_std = stat.stdev(data["pmra"])
    dec_std = stat.stdev(data["pmdec"])
    plt.xlim((ra_mean - 3*ra_std*z),(ra_mean + 5*ra_std*z))
    plt.ylim((dec_mean - 3*dec_std*z),(dec_mean + 5*dec_std*z))
    plt.gca().add_patch(Ellipse(xy=(ra_mean,dec_mean),width=(2*ra_std*z),height=(2*dec_std*z),fill=False,color="red"))
    plt.show()

    #Get cluster members by removing pm and parallax z-std outliers
    data = getMems(data,z)
    print("Num Members =", data.shape[0])

    #Get intrinsic magnitudes and color excesses from spec types
    data = getSpecParams(data)

    #get cluster metallicity
    FeH = np.nanmean( data["fem_gspspec"] - data["mh_gspspec"] )
    FeH_lower = np.nanmean( data["fem_gspspec_lower"] - data["mh_gspspec_lower"] )
    FeH_upper = np.nanmean( data["fem_gspspec_upper"] - data["mh_gspspec_upper"] )
        #drop intermediary values
    data = data.drop(columns=["fem_gspspec","fem_gspspec_lower","fem_gspspec_upper","mh_gspspec","mh_gspspec_lower","mh_gspspec_upper"])
    print("Fe/H =", figRound(FeH), ", lowerConf =", figRound(FeH_lower), ", upperConf = ", figRound(FeH_upper))

    #gaia g,bp,rp --> Johnson U,B,V
        #Conversions from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T8
    data["V_app"] = data["phot_g_mean_mag"] + .01760 + .006860*(data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"]) + .1732*(data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"])**2
    data["bp-rp"] = data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"]
        #solving a cubic polynomial for apparent colors, picking largest root (formulas from setting f(G_bp-G_rp)=f(color)=G-V)
    data["(B-V)_app"] = data["bp-rp"].map(lambda x: np.max(np.real(np.poly1d([-.001768,-.2297,-.02385,-.01147+.006860*(x)+.1732*(x**2)]).roots)) )
    data["(V-R)_app"] = data["bp-rp"].map(lambda x: np.max(np.real(np.poly1d([0.2225,-1.016,-0.01784,-.00509+.006860*(x)+.1732*(x**2)]).roots)) )
        #set errors (approx. per conversion formulas)
    data["err_V_app"] = .05
    data["err_(B-V)_app"] = .08
    data["err_(V-R)_app"] = .07
    
    #Plot color-color
        #gaia colors
    plt.figure()
    plt.title("Color-Color (not dereddened)")
    plt.scatter(data["phot_g_mean_mag"] - data["phot_bp_mean_mag"],data["phot_bp_mean_mag"] - data["phot_rp_mean_mag"])
    plt.xlabel("(G-BP)_app [mag]")
    plt.ylabel("(BP-RP)_app [mag]")
    plt.show()
        #johnson colors
    plt.figure()
    plt.title("Color-Color (not dereddened)")
    plt.scatter(data["(B-V)_app"],data["(V-R)_app"])
    plt.xlabel("(B-V)_app [mag]")
    plt.ylabel("(V-R)_app [mag]")
    plt.show()
        #drop gaia magnitudes
    data = data.drop(columns=["phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag","bp-rp"])

    #Format Data For Ease of Use
    data["dMod_app"] = data["V_app"] - data["V_intr"]
    data["err_dMod_app"] = np.sqrt(data["err_V_app"]**2 + data["V_intr"]**2)
    data["colorExcess"] = data["(B-V)_app"] - data["(B-V)_intr"]
    data["err_colorExcess"] = np.sqrt(data["err_(B-V)_app"]**2 + data["err_(B-V)_intr"]**2)

    #Plot Cluster Trig Parallax distances
    plt.figure()
    plt.title("Trig Parallax Distances (Cluster Only)")
    plt.xlabel("Distance (pc)")
    plt.ylabel("Freq")
    plt.hist((data["dist_trig_parallax"]), bins=100, range=(0,1000))
    plt.show()

    #Plot cluster ra, dec
    plt.figure()
    plt.title("Ra, Dec of Cluster Stars")
    plt.scatter(data["ra"],data["dec"])
    plt.xlabel("Ra [degrees]")
    plt.ylabel("Dec [degrees]")
    plt.show()

    #Plot Observed CMD
    plt.figure()
    plt.gca().invert_yaxis()
    plt.title("Observed CMD")
    plt.xlabel("(B-V)_app [Mag]")
    plt.ylabel("V_app [Mag]")
    plt.scatter(data["(B-V)_app"],data["V_app"], color="red", label = "Hyades Stars")
    plt.legend()
    plt.show()

    #Apply Variable Extinction Method; correct for extinction and reddening
    R, R_err, dist_ext, err_dist_ext = varExt(data)
    data = ext_correct(data,R,R_err)

    #Plot corrected CMD
    plt.figure()
    plt.gca().invert_yaxis()
    plt.title("Corrected CMD")
    plt.xlabel("(B-V)_intr [Mag]")
    plt.ylabel("V_rel [Mag]")
    plt.scatter(data["(B-V)_intr"],data["V_rel"], color="red", label = "Hyades Stars")
    plt.legend()
    plt.show()


    #Get Distance Estimates and print them
        #Trig parallax
    dist_trigParallax = stat.mean(data["dist_trig_parallax"])
    err_dist_trigParallax = np.sqrt(np.sum((data["err_dist_trig_parallax"]**2))) / data.shape[0]
        #Spec Parallax
    dist_specParallax = stat.mean(((data["V_rel"] - data["V_intr"] + 5)/5)**10)
    err_dist_specParallax = (1/5)* np.log(10) * np.sqrt(np.sum( (((data["V_rel"] - data["V_intr"] + 5)/5)**10)**2 * (data["err_V_rel"]**2 + data["err_V_intr"]**2) )) / data.shape[0]
        #print
    print("Trig Parallax: " + str(sciRound(dist_trigParallax,err_dist_trigParallax)) + " pc")
    print("Extinction: " + str(sciRound(dist_ext,err_dist_ext)) + " pc")
    print("Spec Parallax " + str(sciRound(dist_specParallax,err_dist_specParallax)) + " pc")



#Returns pandas dataframe by removing z-std outliers (for proper motion and parallax)
def getMems(stars,z):
    stars = stars[(np.abs(stats_sci.zscore(stars["pmra"]))<z) & (np.abs(stats_sci.zscore(stars["pmdec"]))<z) & (np.abs(stats_sci.zscore(stars["dist_trig_parallax"]))<z)]
    return stars

#Applies Variable Extinction Method and returns R value and distance estimate
def varExt(stars):
        #Regresson to get R
        #errors estimated by taking sigma=sqrt(err_dMod_app**2 + (3.1)**2 * err_colorExcess**2)
    popt,pcov = curve_fit(ext_fit, stars["colorExcess"], stars["dMod_app"],sigma=np.sqrt(stars["err_dMod_app"]**2 + (3.1)**2 * stars["err_colorExcess"]**2))
        #get fit parameters
    perr = np.sqrt(np.diag(pcov))
    r = popt[0]
    r_err = perr[0]
    dMod_real = popt[1]
    dMod_real_err = perr[1]
        #Calculate distance estimate with error
    dMod = ufloat(dMod_real,dMod_real_err)
    dist = 10**((dMod + 5) / 5)
    dist_err = dist.std_dev
    dist = dist.nominal_value

    #Print Results
    print("R = " + str(sciRound(r,r_err)))
        #Use R=3.1 if regression R values doesn't make sense
    if r<0 or r>10:
        r,r_err = 3.1,.1
        print("Assuming R=3.1 +/- .1")

    #Create variable Extinciton graph
    plt.figure()
    plt.title("Variable Extinction (Non-Reddened Removed from R calculation)")
    plt.xlabel("E(B-V) [Mag]")
    plt.ylabel("dMod_app [Mag]")
    plt.scatter(stars["colorExcess"],stars["dMod_app"], color="red", label = "Hyades Stars")
    es = np.linspace(np.min(stars["colorExcess"]),np.max(stars["colorExcess"]),10000)
    plt.plot(es,(dMod_real + es*r), color="blue", label = "Variable Extinction Trend")
    plt.legend()
    plt.show()

    #Return R and extinction distance estimate
    return r, r_err, dist, dist_err

#Returns pandas dataframe to correct for extinction and reddening
def ext_correct(stars,r,r_err):
    #Use intrinsic color from now on to account for reddening
    stars["V_rel"] = stars["V_app"] - r*stars["colorExcess"]
    stars["err_V_rel"] = np.sqrt(stars["err_V_app"]**2 + r**2 * stars["err_colorExcess"]**2)
    return stars
    
#Equation for variable extinction regression
def ext_fit(e, r, dMod_real):
    dMod_app = dMod_real + r*e
    return dMod_app

#round num to 2 sig figs
def figRound(num):
    return float("%.2g" % num)

#get number decimals
def numDec(num):
    if num == 0:
        return 2
    else:
        return -int(np.floor(np.log10(abs(num)))) + 1

#do value and error rounding (to 2 sig figs in error)
def sciRound(num,err):
    err = figRound(err)
    num = round(num, numDec(err))
    return [num,err]

#match gaia and simbad data by RA, Dec
def coordMatch(gaia, simbad):
        #merge_asof needs sorted dataframes
    gaia = gaia.sort_values(by="ra")
    simbad = simbad.sort_values(by="ra")
        #get join tolerance from average ra,dec error in gaia
        #sqrt 2 factor approximates additon of error from simbad
    tolRa = np.abs( np.nanmean(gaia["ra_error"])) / 2 * np.sqrt(2)
    tolDec = np.abs( np.nanmean(gaia["dec_error"])) / 2 * np.sqrt(2)
        #merge_asof only allows merge on one columns, so drop collumns that are above error for dec
    matches = pd.merge_asof(gaia,simbad,on='ra',tolerance=tolRa,direction='nearest')
    matches = matches[np.abs(matches["dec"] - matches["Dec"]) <= tolDec]
        #renaming merged columns
    matches = matches.rename(columns={'ra_x':'ra'})
    if matches.shape[0] == 0:
        print("ERROR: No Coordinate Matches")
        sys.exit()
    return matches

#Get intrinsic color and magnitude from spectral type
    #Use Tsvetkov's tabulated values (https://link.springer.com/content/pdf/10.1134/S1063773708010039.pdf)
def getSpecParams(frame):
        #tabulated data in form {spec type: [V, B-V]}
    tabV = {"O5":-5.6,"O9":-4.5,"B0":-4.0,"B5":-4.2,"A0":0.6,"A5":1.9,"F0":2.7,"F5":3.5,"G0":4.4,"G5":5.1,"K0":5.9,"K5":7.3,"M0":8.8,"M5":12.3}
    tabB_V = {"O5":-0.33,"O9":-0.31,"B0":-0.30,"B5":-0.17,"A0":-0.02,"A5":0.15,"F0":0.30,"F5":0.44,"G0":0.58,"G5":0.68,"K0":0.81,"K5":1.15,"M0":1.40,"M5":1.64}
    order = "OBAFGKM_"
    order2 = ["O5","O5","O9","B0","B5","A0","A5","F0","F5","G0","G5","K0","K5","M0","M5","M5"]
        #deal with different O-type numbers in tabulated data
    frame["spec_type"] = frame["spec_type"].map(lambda x: "O" + str(float(x[1:-1])+5) + "V" if x[:1]=="O" else x)
        #spec types round to nearest tabulated type
    frame["spec_type"] = frame["spec_type"].map(lambda x: x[:1] + str(5*round(float(x[1:-1])/5)) if (5*round(float(x[1:-1])/5) < 10) else order[order.index(x[:1])+1] + "0")
        #deal with M5 being last listed type
    frame["spec_type"] = frame["spec_type"].map(lambda x: "M5" if (x=="_0") else x)
        #get intrinsic/tabulated values
        #estimate error as average half-difference to adjacent bins
    frame["V_intr"] = frame["spec_type"].map(lambda x: tabV[x])
    frame["err_V_intr"] = frame["spec_type"].map(lambda x: np.abs(tabV[order2[order2.index(x)-1]]) + np.abs(tabV[order2[order2.index(x)+1]])/2 if (order2.index(x) != 0) else np.abs(tabV[order2[order2.index(x)+1]])/2)
    frame["(B-V)_intr"] = frame["spec_type"].map(lambda x: tabB_V[x])
    frame["err_(B-V)_intr"] = frame["spec_type"].map(lambda x: np.abs(tabB_V[order2[order2.index(x)-1]]) + np.abs(tabB_V[order2[order2.index(x)+1]])/2 if (order2.index(x) != 0) else np.abs(tabB_V[order2[order2.index(x)+1]])/2)
        #drop spec types
    frame = frame.drop(columns=["spec_type"])
    return frame

#Run Main
main()
