import matplotlib.pyplot as plt
import numpy as np

baseline = 0.0336857286421176
only_unpaired = 0.020579816522603562

data = """cv-10/output.txt:prior: 0.026693591458040157
cv-10/output.txt:prior: 0.028232950485868363
cv-10/output.txt:prior: 0.027219320968909597
cv-10/output.txt:prior: 0.025780729811976436
cv-10/output.txt:prior: 0.024793536756431055
cv-10/output.txt:prior: 0.025088597015843928
cv-10/output.txt:prior: 0.029430680675320516
cv-10/output.txt:prior: 0.026785792517091487
cv-10/output.txt:prior: 0.026492833880693407
cv-10/output.txt:prior: 0.03075194195904232
cv10/output.txt:prior: 0.018105220582142565
cv10/output.txt:prior: 0.016770827267097006
cv10/output.txt:prior: 0.017341437019208462
cv10/output.txt:prior: 0.01934749793171672
cv10/output.txt:prior: 0.019693365535745095
cv10/output.txt:prior: 0.017803232095045946
cv10/output.txt:prior: 0.01971883497606201
cv10/output.txt:prior: 0.01952269885704438
cv10/output.txt:prior: 0.01864821560853491
cv10/output.txt:prior: 0.0183600925386798
cv-1.5/output.txt:prior: 0.020734982334922162
cv-1.5/output.txt:prior: 0.01658414675437048
cv-1.5/output.txt:prior: 0.018085993746690204
cv1.5/output.txt:prior: 0.014291381616805838
cv1.5/output.txt:prior: 0.01553536056666005
cv1.5/output.txt:prior: 0.0141747980796713
cv-2/output.txt:prior: 0.015248078139094144
cv-2/output.txt:prior: 0.01755459032388955
cv2/output.txt:prior: 0.015238429856484147
cv2/output.txt:prior: 0.013993828843467665
cv-3/output.txt:prior: 0.018883668946847086
cv-3/output.txt:prior: 0.02230842412445021
cv-3/output.txt:prior: 0.020615907772577692
cv3/output.txt:prior: 0.01576076373088295
cv3/output.txt:prior: 0.017712319910167387
cv3/output.txt:prior: 0.016565884931692136
cv-4/output.txt:prior: 0.022331133833707797
cv-4/output.txt:prior: 0.0215271030631531
cv-4/output.txt:prior: 0.018818623965880972
cv-4/output.txt:prior: 0.025815296497805843
cv4/output.txt:prior: 0.018723729879283457
cv4/output.txt:prior: 0.016899045592593927
cv4/output.txt:prior: 0.018503173669700106
cv4/output.txt:prior: 0.016255525698843563
cv-5/output.txt:prior: 0.025918893633701076
cv-5/output.txt:prior: 0.022478079348705778
cv-5/output.txt:prior: 0.022313406001329323
cv-5/output.txt:prior: 0.02943246053079181
cv-5/output.txt:prior: 0.02471798395008261
cv5/output.txt:prior: 0.0177482445524099
cv5/output.txt:prior: 0.018322920621559786
cv5/output.txt:prior: 0.017172717781091952
cv5/output.txt:prior: 0.017097674022052468
cv5/output.txt:prior: 0.01595740184108441
cv-6/output.txt:prior: 0.024371996478473268
cv-6/output.txt:prior: 0.024454359180925114
cv-6/output.txt:prior: 0.02399641738738751
cv-6/output.txt:prior: 0.02302977181263585
cv-6/output.txt:prior: 0.031641068749701524
cv-6/output.txt:prior: 0.025317256345072963
cv6/output.txt:prior: 0.017241716200059737
cv6/output.txt:prior: 0.018693463140718552
cv6/output.txt:prior: 0.01764388352722632
cv6/output.txt:prior: 0.0173901098631058
cv6/output.txt:prior: 0.018101910281140922
cv6/output.txt:prior: 0.019560241059885578
cv-7/output.txt:prior: 0.029604574931817618
cv-7/output.txt:prior: 0.02575303925603229
cv-7/output.txt:prior: 0.02300472469961385
cv-7/output.txt:prior: 0.024104940988586846
cv-7/output.txt:prior: 0.025759948917506516
cv-7/output.txt:prior: 0.029337831101769248
cv-7/output.txt:prior: 0.02683779784383201
cv7/output.txt:prior: 0.018883003004374548
cv7/output.txt:prior: 0.017626132949678706
cv7/output.txt:prior: 0.019007805321786028
cv7/output.txt:prior: 0.017632281210684424
cv7/output.txt:prior: 0.01814192617225294
cv7/output.txt:prior: 0.018339243342116156
cv7/output.txt:prior: 0.01895364370353406"""

def parse_data(data, unpaired=True):
    cvdata = {}
    for l in data.split("\n"):
        s, v = l.split("prior:")
        v = float(v)
        s = s.split("/")[0]
        s = float(s.replace("cv", ""))

        if unpaired:
            include = s>0
        else:
            include = s<0

        if include:
            if abs(s) not in cvdata:
                cvdata[abs(s)] = [v]
            else:
                cvdata[abs(s)].append(v)
    print(cvdata)
    pltdata = [(1/k, np.mean(v), np.mean(v)-min(v), max(v)-np.mean(v),) for k, v in cvdata.items()]
    pltdata = sorted(pltdata, key=lambda x: x[0])
    return pltdata

def plotcv(data, unpaired=True, fmt="ro:", label="with unpaired"):

    pltdata = parse_data(data, unpaired=unpaired)
    x = [xye[0] for xye in pltdata]
    y = [xye[1] for xye in pltdata]
    dylower = [xye[2] for xye in pltdata]
    dyupper = [xye[3] for xye in pltdata]
    # plt.plot(x, y, "ro:", label="with unpaired")
    markers, caps, bars = plt.errorbar(x, y, yerr=[dylower, dyupper], fmt=fmt, label=label, capthick=0.5, capsize=2, elinewidth=1)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    return x, y, dylower, dyupper

x, y, _, _ = plotcv(data, unpaired=True, fmt="ro:", label="with unpaired")
x, y, _, _ = plotcv(data, unpaired=False, fmt="bx--", label="without unpaired")

xmin = min(x) - 0.05
xmax = max(x) + 0.05

plt.hlines(y=baseline, xmin=xmin, xmax=xmax, colors="k", linestyles="-", label="identity baseline", lw=2)
plt.hlines(y=only_unpaired, xmin=xmin, xmax=xmax, colors="r", linestyles="-.", label="only unpaired", lw=2)
plt.xlim([xmin, xmax])
plt.ylabel(r"mean $\rm{\Delta (C, C')}$")
plt.xlabel("Proportion of paired data in training")
plt.legend(bbox_to_anchor=[0.5, 1.08], loc='center', ncol=2)
plt.savefig("plt.png")
