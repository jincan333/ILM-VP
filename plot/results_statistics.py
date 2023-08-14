import pandas as pd
import statistics

# inputs=[
#  r'2.13\2.93,2.13\2.13,2.13\6.44'
# ,r'2.13\20.53,2.13\21.01,2.13\22.34'
# ,r'2.13\11.49,2.13\11.49,2.13\12.71'

# ]
# inputs = [
#  r'0.99\2.13,0.99\0.99,0.99\1.51'
# ,r'1.50\20.71,1.62\20.91,0.99\20.80'
# ,r'0.99\11.18,0.99\11.75,0.99\11.20'
# ,r'0.99\0.99,0.99\1.47,0.99\1.33'
# ,r'0.99\3.11,0.99\4.21,0.99\3.22'
# ,r'0.99\14.19,0.99\14.53,0.99\13.32'
# ,r'8.63\8.30,9.90\10.75,9.83\13.10'
# ,r'16.65\25.63,14.67\24.46,15.84\24.99'
# ]
inputs=[r'8.61,7.71,8.65']
for input in inputs:
    temp = []
    for i in input.split(','):
        temp.extend([float(x) for x in i.split('\\')])
    # before_ff = [temp[0],temp[2],temp[4]]
    # after_ff = [temp[1],temp[3],temp[5]]
    # print(
    #     f'{statistics.mean(before_ff):.2f}({statistics.stdev(before_ff):.2f})\{statistics.mean(after_ff):.2f}({statistics.stdev(after_ff):.2f})'
    # )
    after_ff = [temp[0],temp[1],temp[2]]
    print(
        f'{statistics.mean(after_ff):.2f}({statistics.stdev(after_ff):.2f})'
    )