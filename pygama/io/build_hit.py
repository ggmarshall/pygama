import numpy as np
import os,json
import pathlib
import pygama.lh5 as lh5
import pygama.analysis.peak_fitting as pgf
import pygama.pargen.cuts as cut
import pygama.pargen.aoe_cal as cae
import time


def build_hit(f_dsp:str,db_dict:dict, f_hit:str=None,  
                copy_cols:list=None, builder_config:dict=None, overwrite:bool=True):

    t_start = time.time()

    if builder_config is None:
        builder_config = {      
        'energy_param' : 'cuspEmax_ctc', 
        'current_param' : "A_max", 
        'aoe_energy_param' : "cuspEmax",   #Energy param to use for A/E 
        "cut_parameters":{"bl_mean":4, "bl_std":4, "pz_std":4},
        'copy_cols':copy_cols
        }
    pars = [builder_config['energy_param'], builder_config['current_param'], builder_config['aoe_energy_param']]

    if builder_config['copy_cols'] is not None:
        pars = pars + builder_config['copy_cols']

    data = lh5.load_dfs(f_dsp, pars, "raw")
    pass_data, cut_data, mask = cut.load_nda_with_cuts(f_dsp, "raw", [builder_config['energy_param']], 
                                                    cut_parameters =builder_config["cut_parameters"],  return_mask=True)
    data['Passed_quality_cut'] = mask
    data["Cal_energy"] = pgf.poly(data[builder_config['energy_param']], db_dict['ecal_pars'])
    data["A/E"] = data[builder_config['current_param']]/pgf.poly(data[builder_config['aoe_energy_param']], db_dict['ecal_pars'])
    data["A/E"] = cae.get_classifier(data["A/E"], data[builder_config['energy_param']], db_dict['aoe_mu_pars'], db_dict['aoe_sigma_pars'])
    data['Passed_A/E_cut'] = (data["A/E"]>db_dict['aoe_cut_low'])&(data["A/E"]<db_dict['aoe_cut_high'])
    
    if f_hit is None:
        return data

    print('Writing to file:', f_hit)
    if os.path.exists(f_hit) and overwrite==True:
        os.remove(f_hit)

    out_cols = ['A/E', "Cal_energy", 'Passed_quality_cut', 'Passed_A/E_cut']
    if builder_config['copy_cols'] is not None:
        out_cols = out_cols + builder_config['copy_cols']
    sto = lh5.Store()
    col_dict = {col : lh5.Array(data[col].values, attrs={'units':''}) for col in out_cols}
    tb_hit = lh5.Table(size=len(data), col_dict=col_dict)
    tb_name = 'hit'
    sto.write_object(tb_hit, tb_name, f_hit)

    t_elap = time.time() - t_start
    print(f'Done!  Time elapsed: {t_elap:.2f} sec.')