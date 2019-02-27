import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as scs


########################################################################
# This fuction interates all over the file (inputfilename) lines
# It get the number stored a the colonne col_num
# It build and return a tuple made of these numbers
def build_table(inputfilename, col_num=0):
    return tuple(float(i.split()[col_num]) for i in open(inputfilename).readlines()[1:])


def build_sqrt_table(inputfilename, col_num=0):
    return tuple(math.sqrt(float(i.split()[col_num])) for i in open(inputfilename).readlines()[1:])


def reverse_array(array):
    return array[::-1]


########################################################################

def build_show_curves(vect_x, hist_name, ylab, xlab):
    fig = plt.figure(1)
    hist, bins = np.histogram(vect_x[0], bins=50)  # ,normed = True)
    center = (bins[:-1] + bins[1:]) / 2
    p1 = plt.plot(center, hist, color="black")

    hist, bins = np.histogram(vect_x[1], bins=50)  # , normed = True)
    center = (bins[:-1] + bins[1:]) / 2
    p2 = plt.plot(center, hist, color="grey")

    # hist, bins = np.histogram(vect_x[2],bins = 50,range=(float(0.00), float(50.00)))#, normed = True)
    # center = (bins[:-1]+bins[1:])/2
    # p3=plt.plot(center, hist,'--', color = "grey")

    # hist, bins = np.histogram(vect_x[3],bins = 50,range=(float(0.00), float(50.00)))#, normed = True)
    # center = (bins[:-1]+bins[1:])/2
    # p4=plt.plot(center, hist,'--', color = "black")

    # plt.title("Impact of the growth process on fracture length distributions")
    # plt.legend([p1, p2], ["seq", "rand"], loc = 3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)

    plt.savefig(hist_name)


########################################################################

def build_richter_plot(x, filename, point_type="", linewidth="1", color="black", label_data="label", min_bound=50,
                       max_bound=0):
    hist, bins = np.histogram(x, bins=50, range=(float(-5.00), float(0.00)))  # ,normed = True)
    center = (bins[:-1] + bins[1:]) / 2
    t = np.cumsum(reverse_array(hist))
    t = reverse_array(t)
    t = t.astype(float) / float(len(x))
    t = np.log10(t)

    if min_bound == 50:
        plt.plot(center, t, point_type, color=color, linewidth=linewidth)  # , label=label_data )
    else:
        # print regression model
        (aCoeff, bCoeff, rVal, pVal, stdError) = scs.linregress(center[min_bound:max_bound:1], t[min_bound:max_bound:1])
        model = (center[(min_bound - 5):(max_bound + 5):1] * aCoeff) + bCoeff
        plt.plot(center[(min_bound - 5):(max_bound + 5):1], model, color=color, linewidth=1)

        plt.plot(center, t, point_type, color=color, linewidth=linewidth,
                 label=label_data + ';  $b=%10.2f$' % (-aCoeff))
        fichier = open(filename, "a")
        fichier.write('%10.2f %10.2f\n' % (-aCoeff, stdError))
        fichier.close()


# dmin = 999999.
# min_bound = 50
# max_bound = 44
#
# for reg1 in range(5,35) :
#   	(a1Coeff,b1Coeff,r1Val,p1Val,std1Error) = scs.linregress(center[:reg1],t[:reg1])
# 	print r1Val**2
#   	if r1Val**2>0.4 :
# 	  	min_bound = reg1

#	it = 50 - min_bound
#	prev_rval = 0.
#	for reg2 in range (1,it) :
#		(a2Coeff,b2Coeff,r2Val,p2Val,std2Error) = scs.linregress(center[reg2:(50-reg2)],t[reg2:(50-reg2)])
#		print r2Val**2
#		if (r2Val**2-prev_rval)<0.4 :
#			max_bound = 50-reg2
#		prev_rval=r2Val**2

#	print(min_bound, max_bound)


#  for min in range(25,30) :
#	  for max in range(min+10, 48)  :
#		  if (min < max ) :
#			  (a1Coeff,b1Coeff,r1Val,p1Val,std1Error) = scs.linregress(center[:min],t[:min])
#			  (a2Coeff,b2Coeff,r2Val,p2Val,std2Error) = scs.linregress(center[min:max:1],t[min:max:1])
#			  (a3Coeff,b3Coeff,r3Val,p3Val,std3Error) = scs.linregress(center[max:],t[max:])
#			  d_tmp = r1Val**2+r2Val**2+r3Val**2
#			  if (dmin>d_tmp):
#				  dmin = d_tmp
#				  min_bound = min
#				  max_bound = max
#				  print(min,max,dmin)


def build_omori_plot_figure(x, y, graph_name, filename):
    xmin = min(x)
    xmax = max(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(xmin, xmax)
    lm = ax1.loglog(x, y, '+', color="black", label=r'Measures $\frac{N(\delta t)}{\delta t}$', markersize=15)

    # print regression model
    (aCoeff, bCoeff, rVal, pVal, stdError) = scs.linregress(np.log10(x[4:11]), np.log10(y[4:11]))
    t = x
    # t = np.logspace(math.log(xmin),math.log(xmax))
    model = 10 ** (bCoeff) * (t ** aCoeff)
    lr = plt.loglog(t, model, color="black", label=r'$\frac{N(\delta t)}{\delta t}\ \sim \ \delta t^{-p}$', linewidth=3)

    # print dimension fractal
    ax2 = plt.twinx()

    dim_f = []
    for i in t:
        dim_f.append(aCoeff)
    df = ax2.plot(t, dim_f, "--", color="grey", label=r'$p=%10.2f\ \pm \ %10.2f$' % (aCoeff, stdError), linewidth=3)
    # ax2.set_ylim(2.0, 4.0)
    ax2.set_xlim(xmin, xmax)

    fichier = open(filename, "a")
    fichier.write('%10.2f %10.2f\n' % (aCoeff, stdError))
    fichier.close()

    # print uncertainty
    #	uncertainty = [];
    #	x2=[]
    #	for i in range(0,len(x)-2) :
    #		uncertainty.append((np.log10(y[i+1])-np.log10(y[i]))/(np.log10(x[i+1])-np.log10(x[i])))
    #		x2.append(x[i]+((x[i+1]-x[i])/2))
    #	lres = ax2.plot(x2,uncertainty, ':.', color = "grey",label = 'Residual $p$',linewidth=3, markersize=12)

    # settings
    ax1.set_xlabel(r"$\delta t$", fontsize=20)
    ax1.set_ylabel(r"$\frac{N(\delta t)}{\delta t}$", fontsize=20)
    ax2.set_ylabel(
        r"$p=\frac{\delta\ \log\left(\frac{N(\delta t)}{\delta t}\right)}{\delta\ \log\left(\delta t\right)}$",
        fontsize=20)
    lns = lm + lr + df  # +lres
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)
    #	ax1.set_ylim(0.0000001, 0.1)
    fig.tight_layout()
    #	plt.show()
    plt.savefig(graph_name)


# plt.savefig(graph_name)


def build_fractal_dimension_figure(x, y, graph_name, filename):
    xmin = min(x)
    xmax = max(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(xmin, xmax)
    lm = ax1.loglog(x, y, '+', color="black", label='Measures $C_{2}(r)$', markersize=15)

    # print regression model
    (aCoeff, bCoeff, rVal, pVal, stdError) = scs.linregress(np.log10(x[4:]), np.log10(y[4:]))
    t = x
    # t = np.logspace(math.log(xmin),math.log(xmax))
    model = 10 ** (bCoeff) * (t ** aCoeff)
    lr = plt.loglog(t, model, color="black", label='$C_{2}(r)\ \sim \ r^{D_c}$', linewidth=3)

    # print dimension fractal
    ax2 = plt.twinx()

    dim_f = []
    for i in t:
        dim_f.append(aCoeff)
    df = ax2.plot(t, dim_f, "--", color="grey", label='$D_c=%10.2f\ \pm \ %10.2f$' % (aCoeff, stdError), linewidth=3)
    #   ax2.set_ylim(2.0, 4.0)
    ax2.set_xlim(xmin, xmax)

    fichier = open(filename, "a")
    fichier.write('%10.2f %10.2f\n' % (aCoeff, stdError))
    fichier.close()

    # print uncertainty
    uncertainty = [];
    x2 = []
    for i in range(0, len(x) - 2):
        uncertainty.append((np.log10(y[i + 1]) - np.log10(y[i])) / (np.log10(x[i + 1]) - np.log10(x[i])))
        x2.append(x[i] + ((x[i + 1] - x[i]) / 2))
    lres = ax2.plot(x2, uncertainty, ':.', color="grey", label='Residual $D_c$', linewidth=3, markersize=12)

    # settings
    ax1.set_xlabel(r"$r$", fontsize=20)
    ax1.set_ylabel(r"$C_{2}\left(r\right)$", fontsize=20)
    ax2.set_ylabel(r"$D_c=\frac{\delta\ \log\left(C_{2}\left(r\right)\right)}{\delta\ \log\left(r\right)}$",
                   fontsize=20)
    lns = lm + lr + df  # +lres
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)
    #	ax1.set_ylim(0.0000001, 0.1)
    fig.tight_layout()
    #	plt.show()
    plt.savefig(graph_name)
# plt.savefig(graph_name)
