import pandas as pd
import statistics


inputs=[
 r'1.01\53.36,0.87\54.21,0.85\52.27'
]
# inputs = [
#  r'2.24\66.07,2.55\65.72,1.69\65.89'
# ,r'2.93\78.59,4.57\78.43,3.85\78.48'
# ,r'17.09\95.35,17.17\95.09,15.42\95.09'
# ,r'2.90\77.41,2.86\77.98,2.98\77.24'
# ,r'8.16\95.94,3.98\95.53,6.05\96.26'
# ,r'11.81\67.02,7.39\66.65,6.17\67.29'
# ,r'3.03\82.45,6.81\82.42,4.61\83.21'
# ,r'1.22\58.57,0.85\59.74,0.77\57.27'
# ]
# inputs = [
#  r'10.60\88.20,10.02\88.89,13.09\88.95'
# ,r''
# ,r'3.21\87.41,5.52\88.92,3.57\88.23'
# ,r'3.03\55.21,2.50\54.95,2.93\57.61'
# ]
for input in inputs:
    temp = []
    if len(input) == 0:
        continue
    for i in input.split(','):
        temp.extend([float(x) for x in i.split('\\')])
    # before_ff = [temp[0],temp[2],temp[4]]
    # after_ff = [temp[1],temp[3],temp[5]]
    # print(
    #     f'{statistics.mean(before_ff):.2f}({statistics.stdev(before_ff):.2f})\{statistics.mean(after_ff):.2f}({statistics.stdev(after_ff):.2f})'
    # )
    if len(temp) == 6:
        after_ff = [temp[1],temp[3],temp[5]]
    elif len(temp) == 3:
        after_ff = [temp[0],temp[1],temp[2]]
    else:
        after_ff = [0, 0, 0]
    print(
        f'{statistics.mean(after_ff):.2f}({statistics.stdev(after_ff):.2f})'
    )