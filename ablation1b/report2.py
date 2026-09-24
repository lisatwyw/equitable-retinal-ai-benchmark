 
def format_metric2(value, ci):
    return f"{value:.4f}  [{ci[0]:.2f} – {ci[1]:.2f}]"      

  

for k in metrics.keys():
    S=list(metrics[k].keys())[1:]
    print(  [ '%s=%.2f'%( kk, metrics[k][kk] ) for kk in S ],k, '\n', ) 

try:
    metrics_to_print = [
        "AUROC", "AUPRC", "Accuracy", "Balanced Accuracy",
        "F1", "F1 Macro", "Precision", "Recall", "FPR"
    ]
    
    data = {}
    
    for side in sides:
        for MODE in MODES:
            values = []
            for k in metrics_to_print:
                value = format_metric2(
                    metrics[side, MODE][k],
                    cis[side, MODE][k]
                )
                values.append(value)
            data[(side,MODE.replace('_',' '))] = values
    
    int_res = pd.DataFrame(data, index=[m.replace('_',' ')  for m in metrics_to_print])    
    
    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 100)
    print(int_res.to_string())
    
    for d in [0,1,2]:
        for side in sides:
            for MODE in MODES:        
                values = []
                try:
                    for k in metrics_to_print:
                        value = format_metric2(
                            metrics[side, MODE, DSET, d, 'uncalib'][k],
                            cis[side, MODE, DSET, d, 'uncalib'][k]
                        )
                        value2 = format_metric2(
                            metrics[side, MODE, DSET, d, ][k],
                            cis[side, MODE, DSET, d ][k]
                        )
                        values.append(value + ' | '+ value2)
                    data[(side,MODE.replace('_',' '), d)] = values
                except:
                    pass
                
    
    ext_res = pd.DataFrame(data, index=[m.replace('_',' ')  for m in metrics_to_print])    
    print("\n" + "=" * 100)
    print("EXTERNAL validation (95% C.I.)")
    print("=" * 100)     
    print(ext_res.to_string())
except:
    pass
    
print( int_res.to_latex() )
print( ext_res.to_latex() )

'''
latex = df.to_latex(
    multicolumn=True,
    multirow=True,
    multicolumn_format="c",
    escape=False,
    index=True,
    hrules=True
)

print(latex)
'''



