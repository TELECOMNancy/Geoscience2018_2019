import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import gamma
from sklearn.preprocessing import MinMaxScaler

#df=pd.read_csv(
#'/Users/sakina/Downloads/crack.csv', sep= ' ')


# Analyse temporelle 
# Loi d'Omori 

# Loi omori simplifiée 
def simple_omori(pas_de_temps, t ):
    t_declenchement_microcrack = t * pas_de_temps 
    
    # Normalisation de t
    min = np.min( t_declenchement_microcrack)
    max = np.max(t_declenchement_microcrack)
    t_normalized = t_declenchement_microcrack * len(t_declenchement_microcrack) / (max - min)

    Nt = 1/t_declenchement_microcrack
    #print(t_declenchement_microcrack.head())
    
# Loi omori (Utsu) 
def omori(pas_de_temps, t , K, p_value, c):
    t_declenchement_microcrack = t * pas_de_temps
    Nt = K/np.power(t_declenchement_microcrack + c , p_value)
    print(Nt.head())

    # Normalisation de t
    min = np.min( t_declenchement_microcrack)
    max = np.max(t_declenchement_microcrack)
    t_normalized = t_declenchement_microcrack * len(t_declenchement_microcrack) / (max - min)

# Calcul densité de probabilité (loi Gamma)
def proba(t_normalized):
    b = np.var(t_normalized)/np.mean(t_normalized)
    print("beta " , b)
    gam = np.mean(t_normalized)/b
    print("gamma " , gam)
    return C(gam, b) * np.power(t_normalized, gam - 1) * np.exp(-t_normalized / b)

def C(gam, b):
    C =  np.power(b, gam) * gamma(gam)
    return 1/C

def analyse_temporelle(df):
    t_crack = df['i'] * 0.0001 
    #test_crack= t_crack[:1000]
    l = len(t_crack)
    print(l)
    '''
    dif = np.diff(t_crack)
    # Supprimer les valeurs nulles
    dif = dif[dif!=0]
    # Tri de dif
    dif = np.sort(dif)
    min = np.min(dif)
    max = np.max(dif)
    # Normalisation de t
    lamda = len(dif) / (max - min)
    t_normalized = dif * lamda
    '''
    delta_t = []
    for i in range(l):
        for j in range(l-i):
            delta_t += [t_crack[j+i] - t_crack[i]]
    delta_t = np.array(delta_t)
    delta_t = delta_t[np.nonzero(delta_t)] 
    delta_t = np.sort(delta_t)
    # Normalisation
    t_normalized = delta_t/np.max(delta_t)

    p = proba(t_normalized)
    
    

    fig, ax = plt.subplots()
    plt.plot(  t_normalized, p)
    plt.plot(  t_normalized, 1/t_normalized)
    #plt.plot(  t_normalized, np.exp(-t_normalized))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Temps normalisé')
    plt.ylabel('Densité de probabilité temporelle')
    plt.grid()
    plt.show() 
    
"""
def build_omori_plot_figure(x, y, graph_name, filename) :
	xmin=min(x)
	xmax=max(x)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_xlim(xmin,xmax)
	lm = ax1.loglog (x, y, '+',color = "black", label = r'Measures $\frac{N(\delta t)}{\delta t}$',markersize=15)

	#print regression model
	(aCoeff,bCoeff,rVal,pVal,stdError) = scs.linregress(np.log10(x[4:11]),np.log10(y[4:11]))
	t=x
	#t = np.logspace(math.log(xmin),math.log(xmax))
	model = 10**(bCoeff)*(t**aCoeff)
	lr = plt.loglog(t, model, color = "black",  label = r'$\frac{N(\delta t)}{\delta t}\ \sim \ \delta t^{-p}$',linewidth =3 )

	#print dimension fractal
	ax2 = plt.twinx()

	dim_f = []
	for i in t :
		dim_f.append(aCoeff)
	df = ax2.plot(t,dim_f, "--",color = "grey", label = r'$p=%10.2f\ \pm \ %10.2f$' %(aCoeff,stdError),linewidth=3 )
	#ax2.set_ylim(2.0, 4.0)
	ax2.set_xlim(xmin,xmax)

	fichier = open(filename, "a")
	fichier.write('%10.2f %10.2f\n' %(aCoeff,stdError) )
	fichier.close()

	#print uncertainty
#	uncertainty = [];
#	x2=[]
#	for i in range(0,len(x)-2) :
#		uncertainty.append((np.log10(y[i+1])-np.log10(y[i]))/(np.log10(x[i+1])-np.log10(x[i])))
#		x2.append(x[i]+((x[i+1]-x[i])/2))
#	lres = ax2.plot(x2,uncertainty, ':.', color = "grey",label = 'Residual $p$',linewidth=3, markersize=12)


	#settings
	ax1.set_xlabel(r"$\delta t$" , fontsize=20)
	ax1.set_ylabel(r"$\frac{N(\delta t)}{\delta t}$", fontsize=20)
	ax2.set_ylabel(r"$p=\frac{\delta\ \log\left(\frac{N(\delta t)}{\delta t}\right)}{\delta\ \log\left(\delta t\right)}$",fontsize=20)
	lns = lm+lr+df#+lres
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs,  loc=4)
#	ax1.set_ylim(0.0000001, 0.1)
	fig.tight_layout()
#	plt.show()
	plt.savefig(graph_name)
	#plt.savefig(graph_name) '''
	