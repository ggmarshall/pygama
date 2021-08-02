Search.setIndex({docnames:["index","install","modules","pygama","pygama.analysis","pygama.dsp","pygama.io","pygama.lh5"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","install.rst","modules.rst","pygama.rst","pygama.analysis.rst","pygama.dsp.rst","pygama.io.rst","pygama.lh5.rst"],objects:{"":{pygama:[3,0,0,"-"]},"pygama.analysis":{calibration:[4,0,0,"-"],data_cleaning:[4,0,0,"-"],datagroup:[4,0,0,"-"],histograms:[4,0,0,"-"],metadata:[4,0,0,"-"],peak_fitting:[4,0,0,"-"]},"pygama.analysis.calibration":{calibrate_tl208:[4,1,1,""],get_calibration_energies:[4,1,1,""],get_hpge_E_peak_bounds:[4,1,1,""],get_hpge_E_peak_par_guess:[4,1,1,""],get_i_local_extrema:[4,1,1,""],get_i_local_maxima:[4,1,1,""],get_i_local_minima:[4,1,1,""],get_most_prominent_peaks:[4,1,1,""],hpge_E_calibration:[4,1,1,""],hpge_find_E_peaks:[4,1,1,""],hpge_fit_E_cal_func:[4,1,1,""],hpge_fit_E_peak_tops:[4,1,1,""],hpge_fit_E_peaks:[4,1,1,""],hpge_fit_E_scale:[4,1,1,""],hpge_get_E_peaks:[4,1,1,""],match_peaks:[4,1,1,""],poly_match:[4,1,1,""]},"pygama.analysis.data_cleaning":{find_pulser_properties:[4,1,1,""],gaussian_cut:[4,1,1,""],tag_pulsers:[4,1,1,""],xtalball_cut:[4,1,1,""]},"pygama.analysis.datagroup":{DataGroup:[4,2,1,""]},"pygama.analysis.datagroup.DataGroup":{get_lh5_cols:[4,3,1,""],lh5_dir_setup:[4,3,1,""],load_df:[4,3,1,""],load_keys:[4,3,1,""],save_df:[4,3,1,""],save_keys:[4,3,1,""],scan_daq_dir:[4,3,1,""],set_config:[4,3,1,""]},"pygama.analysis.histograms":{better_int_binning:[4,1,1,""],find_bin:[4,1,1,""],get_bin_centers:[4,1,1,""],get_bin_widths:[4,1,1,""],get_fwfm:[4,1,1,""],get_fwhm:[4,1,1,""],get_gaussian_guess:[4,1,1,""],get_hist:[4,1,1,""],plot_hist:[4,1,1,""],range_slice:[4,1,1,""]},"pygama.analysis.metadata":{write_pretty:[4,1,1,""]},"pygama.analysis.peak_fitting":{Am_double:[4,1,1,""],cal_slope:[4,1,1,""],double_gauss:[4,1,1,""],fit_binned:[4,1,1,""],fit_hist:[4,1,1,""],fit_unbinned:[4,1,1,""],gauss:[4,1,1,""],gauss_basic:[4,1,1,""],gauss_bkg:[4,1,1,""],gauss_cdf:[4,1,1,""],gauss_int:[4,1,1,""],gauss_lin:[4,1,1,""],gauss_mode_max:[4,1,1,""],gauss_mode_width_max:[4,1,1,""],gauss_step:[4,1,1,""],gauss_tail:[4,1,1,""],gauss_tail_approx:[4,1,1,""],gauss_tail_exact:[4,1,1,""],get_bin_estimates:[4,1,1,""],get_fwhm_func:[4,1,1,""],get_mu_func:[4,1,1,""],goodness_of_fit:[4,1,1,""],neg_log_like:[4,1,1,""],neg_poisson_log_like:[4,1,1,""],poisson_gof:[4,1,1,""],poly:[4,1,1,""],radford_fwhm:[4,1,1,""],radford_parameter_gradient:[4,1,1,""],radford_peak:[4,1,1,""],radford_peak_wrapped:[4,1,1,""],radford_peakshape_derivative:[4,1,1,""],step:[4,1,1,""],taylor_mode_max:[4,1,1,""],xtalball:[4,1,1,""]},"pygama.dsp":{ProcessingChain:[5,0,0,"-"],build_processing_chain:[5,0,0,"-"],dsp_optimize:[5,0,0,"-"],processors:[5,0,0,"-"],units:[5,0,0,"-"]},"pygama.dsp.ProcessingChain":{ProcessingChain:[5,2,1,""]},"pygama.dsp.ProcessingChain.ProcessingChain":{add_input_buffer:[5,3,1,""],add_output_buffer:[5,3,1,""],add_processor:[5,3,1,""],add_scalar:[5,3,1,""],add_waveform:[5,3,1,""],execute:[5,3,1,""],execute_block:[5,3,1,""],get_input_buffer:[5,3,1,""],get_output_buffer:[5,3,1,""],get_variable:[5,3,1,""]},"pygama.dsp.build_processing_chain":{build_processing_chain:[5,1,1,""]},"pygama.dsp.dsp_optimize":{ParGrid:[5,2,1,""],ParGridDimension:[5,2,1,""],get_grid_points:[5,1,1,""],run_grid:[5,1,1,""],run_grid_multiprocess_parallel:[5,1,1,""],run_grid_point:[5,1,1,""],run_multi_grid:[5,1,1,""],run_one_dsp:[5,1,1,""]},"pygama.dsp.dsp_optimize.ParGrid":{add_dimension:[5,3,1,""],check_indices:[5,3,1,""],get_data:[5,3,1,""],get_n_dimensions:[5,3,1,""],get_n_grid_points:[5,3,1,""],get_n_points_of_dim:[5,3,1,""],get_par_meshgrid:[5,3,1,""],get_shape:[5,3,1,""],get_zero_indices:[5,3,1,""],iterate_indices:[5,3,1,""],print_data:[5,3,1,""],set_dsp_pars:[5,3,1,""]},"pygama.dsp.dsp_optimize.ParGridDimension":{name:[5,4,1,""],parameter:[5,4,1,""],value_strs:[5,4,1,""]},"pygama.io":{ch_group:[6,0,0,"-"],compassdaq:[6,0,0,"-"],daq_to_raw:[6,0,0,"-"],fcdaq:[6,0,0,"-"],io_base:[6,0,0,"-"],lh5:[6,0,0,"-"],llamadaq:[6,0,0,"-"],orca_digitizers:[6,0,0,"-"],orcadaq:[6,0,0,"-"],pollers:[6,0,0,"-"],wfcompress:[6,0,0,"-"]},"pygama.io.ch_group":{build_tables:[6,1,1,""],create_dummy_ch_group:[6,1,1,""],expand_ch_groups:[6,1,1,""],get_list_of:[6,1,1,""],set_outputs:[6,1,1,""]},"pygama.io.compassdaq":{CAENDT57XX:[6,2,1,""],process_compass:[6,1,1,""]},"pygama.io.compassdaq.CAENDT57XX":{assemble_data_row:[6,3,1,""],create_dataframe:[6,3,1,""],get_event:[6,3,1,""],get_event_size:[6,3,1,""],input_config:[6,3,1,""]},"pygama.io.daq_to_raw":{daq_to_raw:[6,1,1,""]},"pygama.io.fcdaq":{FlashCamEventDecoder:[6,2,1,""],FlashCamStatusDecoder:[6,2,1,""],process_flashcam:[6,1,1,""]},"pygama.io.fcdaq.FlashCamEventDecoder":{decode_packet:[6,3,1,""],get_decoded_values:[6,3,1,""],get_file_config_struct:[6,3,1,""],set_file_config:[6,3,1,""]},"pygama.io.fcdaq.FlashCamStatusDecoder":{decode_packet:[6,3,1,""],set_file_config:[6,3,1,""]},"pygama.io.io_base":{DataDecoder:[6,2,1,""],DataTaker:[6,2,1,""]},"pygama.io.io_base.DataDecoder":{get_decoded_values:[6,3,1,""],initialize_lh5_table:[6,3,1,""],put_in_garbage:[6,3,1,""],write_out_garbage:[6,3,1,""]},"pygama.io.llamadaq":{BinaryReadException:[6,5,1,""],LLAMAStruck3316:[6,2,1,""],llama_3316:[6,2,1,""],process_llama_3316:[6,1,1,""]},"pygama.io.llamadaq.BinaryReadException":{printMessage:[6,3,1,""]},"pygama.io.llamadaq.LLAMAStruck3316":{decode_event:[6,3,1,""],initialize:[6,3,1,""],readMetadata:[6,3,1,""]},"pygama.io.llamadaq.llama_3316":{parse_channelConfigs:[6,3,1,""],parse_fileheader:[6,3,1,""],read_next_event:[6,3,1,""]},"pygama.io.orca_digitizers":{ORCAGretina4M:[6,2,1,""],ORCAStruck3302:[6,2,1,""]},"pygama.io.orca_digitizers.ORCAGretina4M":{decode_packet:[6,3,1,""],get_decoded_values:[6,3,1,""],is_multisampled:[6,3,1,""],max_n_rows_per_packet:[6,3,1,""],set_object_info:[6,3,1,""]},"pygama.io.orca_digitizers.ORCAStruck3302":{decode_packet:[6,3,1,""],get_decoded_values:[6,3,1,""],max_n_rows_per_packet:[6,3,1,""],set_object_info:[6,3,1,""]},"pygama.io.orcadaq":{OrcaDecoder:[6,2,1,""],flip_data_ids:[6,1,1,""],from_bytes:[6,1,1,""],get_card:[6,1,1,""],get_ccc:[6,1,1,""],get_channel:[6,1,1,""],get_crate:[6,1,1,""],get_data_id:[6,1,1,""],get_id_to_decoder_name_dict:[6,1,1,""],get_next_packet:[6,1,1,""],get_object_info:[6,1,1,""],get_run_number:[6,1,1,""],open_orca:[6,1,1,""],parse_header:[6,1,1,""],process_orca:[6,1,1,""]},"pygama.io.orcadaq.OrcaDecoder":{set_header_dict:[6,3,1,""],set_object_info:[6,3,1,""]},"pygama.io.pollers":{ISegHVDecoder:[6,2,1,""],MJDPreampDecoder:[6,2,1,""]},"pygama.io.pollers.ISegHVDecoder":{decode_event:[6,3,1,""]},"pygama.io.pollers.MJDPreampDecoder":{decode_event:[6,3,1,""],get_detectors_for_preamp:[6,3,1,""]},"pygama.io.wfcompress":{compression:[6,1,1,""],decompression:[6,1,1,""],empty:[6,1,1,""],nda_to_vect:[6,1,1,""],vect_to_nda:[6,1,1,""]},"pygama.lh5":{array:[7,0,0,"-"],arrayofequalsizedarrays:[7,0,0,"-"],fixedsizearray:[7,0,0,"-"],lh5_utils:[7,0,0,"-"],scalar:[7,0,0,"-"],store:[7,0,0,"-"],struct:[7,0,0,"-"],table:[7,0,0,"-"],vectorofvectors:[7,0,0,"-"]},"pygama.lh5.array":{Array:[7,2,1,""]},"pygama.lh5.array.Array":{__len__:[7,3,1,""],dataype_name:[7,3,1,""],form_datatype:[7,3,1,""],resize:[7,3,1,""]},"pygama.lh5.arrayofequalsizedarrays":{ArrayOfEqualSizedArrays:[7,2,1,""]},"pygama.lh5.arrayofequalsizedarrays.ArrayOfEqualSizedArrays":{__len__:[7,3,1,""],dataype_name:[7,3,1,""],form_datatype:[7,3,1,""]},"pygama.lh5.fixedsizearray":{FixedSizeArray:[7,2,1,""]},"pygama.lh5.fixedsizearray.FixedSizeArray":{dataype_name:[7,3,1,""]},"pygama.lh5.lh5_utils":{get_lh5_element_type:[7,1,1,""],parse_datatype:[7,1,1,""]},"pygama.lh5.scalar":{Scalar:[7,2,1,""]},"pygama.lh5.scalar.Scalar":{datatype_name:[7,3,1,""],form_datatype:[7,3,1,""]},"pygama.lh5.store":{Store:[7,2,1,""],load_dfs:[7,1,1,""],load_nda:[7,1,1,""]},"pygama.lh5.store.Store":{get_buffer:[7,3,1,""],gimme_file:[7,3,1,""],gimme_group:[7,3,1,""],ls:[7,3,1,""],read_n_rows:[7,3,1,""],read_object:[7,3,1,""],write_object:[7,3,1,""]},"pygama.lh5.struct":{Struct:[7,2,1,""]},"pygama.lh5.struct.Struct":{add_field:[7,3,1,""],datatype_name:[7,3,1,""],form_datatype:[7,3,1,""],update_datatype:[7,3,1,""]},"pygama.lh5.table":{Table:[7,2,1,""]},"pygama.lh5.table.Table":{__len__:[7,3,1,""],add_field:[7,3,1,""],clear:[7,3,1,""],datatype_name:[7,3,1,""],get_dataframe:[7,3,1,""],is_full:[7,3,1,""],push_row:[7,3,1,""],resize:[7,3,1,""]},"pygama.lh5.vectorofvectors":{VectorOfVectors:[7,2,1,""]},"pygama.lh5.vectorofvectors.VectorOfVectors":{__len__:[7,3,1,""],datatype_name:[7,3,1,""],form_datatype:[7,3,1,""],resize:[7,3,1,""],set_vector:[7,3,1,""]},"pygama.utils":{SafeDict:[3,2,1,""],fit_simple_scaling:[3,1,1,""],get_dataset_from_cmdline:[3,1,1,""],get_formatted_stats:[3,1,1,""],get_par_names:[3,1,1,""],linear_fit_by_sums:[3,1,1,""],peakdet:[3,1,1,""],plot_func:[3,1,1,""],print_fit_results:[3,1,1,""],set_plot_style:[3,1,1,""],sh:[3,1,1,""],sizeof_fmt:[3,1,1,""],tree_draw:[3,1,1,""],update_progress:[3,1,1,""]},pygama:{analysis:[4,0,0,"-"],dsp:[5,0,0,"-"],io:[6,0,0,"-"],lh5:[7,0,0,"-"],utils:[3,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:exception"},terms:{"0":[4,5,6,7],"05":4,"08":4,"09790931254396479":4,"1":[3,4,5,6,7],"10":4,"100":4,"10000":4,"1024":6,"103kev":4,"10939486522749278":4,"117":6,"133ba":4,"15860757":3,"15919638684132664":4,"16":5,"18":6,"1bbbwk":3,"1d":[4,7],"1e":4,"1st":7,"2":[3,4,5,6],"20":4,"2019":6,"2041666666666666":4,"20document":4,"23":6,"24":6,"241am":4,"250":4,"256":6,"2614":4,"2d":4,"2nd":4,"2x":6,"2x2":4,"3":[4,5,7],"3083363869003466":4,"32":6,"3302":6,"3316":6,"3d":6,"3x3":4,"4":[3,4,6],"467":6,"5":[4,5,6],"50":4,"6":6,"64":6,"683":6,"7":4,"717":6,"75":4,"8":[5,6],"8192":6,"81kev":4,"85":4,"9140":4,"9223372036854775807":7,"99kev":4,"byte":[3,6],"case":4,"class":[3,4,5,6,7],"const":4,"default":[4,5,6,7],"do":[1,4,5,6,7],"enum":6,"final":4,"float":[3,4,5],"function":[3,4,5,6,7],"import":[3,4],"int":[4,5,6,7],"long":[6,7],"new":[4,5,6],"public":3,"return":[3,4,5,6,7],"short":6,"static":5,"super":6,"true":[3,4,5,6,7],"try":5,"var":[3,4],"void":5,"while":7,A:[4,5,6,7],And:3,At:[4,5],But:4,By:5,For:[3,4,5,7],If:[1,3,4,5,6,7],In:[3,5,6],It:5,Or:3,The:[3,4,5,6,7],There:[3,5],These:4,To:[1,5,7],__all__:3,__init__:[3,6],__len__:7,__read_chunk_head:6,__read_next_ev:6,a1:4,a2:4,a3:4,abc:6,abl:6,about:[3,4,5,6],abov:4,absolut:4,accept:5,access:6,accommod:7,accord:[4,6],account:4,accur:4,acquisit:0,act:5,action:3,actual:[6,7],ad:[4,5,6,7],adapt:3,adc:4,add:[3,5,7],add_dimens:5,add_field:7,add_input_buff:5,add_output_buff:5,add_processor:5,add_scalar:5,add_vector:5,add_waveform:5,addit:[1,5],admin:1,advantag:4,after:[4,6,7],algorithm:4,alia:[4,5,7],all:[4,5,6,7],alloc:[4,5,7],allow:6,along:[6,7],alreadi:[4,5],also:[4,5,6,7],alwai:[5,7],am_doubl:4,amax:4,among:4,amount:7,amplitud:4,an:[3,4,5,6,7],an_id:6,analysi:[0,2,3],ani:[4,5,7],answer:3,anywai:4,api:[0,3],app:4,appear:5,appearin:5,append:7,appli:[5,7],applic:7,appropri:[4,6,7],approxiamt:4,approxim:4,ar:[3,4,5,6,7],arbitrari:[4,5],area:4,arg:[3,4,5,6],argmax:4,argpars:3,argument:[3,4,5,7],arithmet:6,around:[3,4],arrai:[0,2,3,4,5,6],arrayofequalsizedarrai:[0,2,3],assembl:6,assemble_data_row:6,associ:[0,5,6,7],assuem:4,assum:4,atol:4,attach:6,attempt:4,attr:[5,7],attribut:[6,7],auto:[3,4],automat:[3,4,5,6],aux:6,avail:[4,5],avoid:4,ax:[5,7],axi:[3,5,7],b1:4,b2:4,b8765:6,b:[3,4,6],background:4,backward:6,bad:4,bar:4,base:[3,4,5,6,7],base_group:7,base_path:7,basic:[5,6],becaus:4,been:5,befor:[6,7],begin:6,behav:4,belong:6,below:7,best:[3,4,6],beta:4,better:[3,4],better_int_bin:4,between:4,bg0:4,big_endian:6,billauer:[3,4],bin:[4,6],bin_cent:4,bin_edges_arrai:4,bin_width:4,binari:[4,5,6],binaryeventdata:6,binaryreadexcept:6,bind:5,bins_over_f:4,binwidth:4,bit:6,bitbank:6,bitshift:6,bitwis:6,bkg:4,bl:4,blank:3,block:5,block_width:5,boiler:6,bool:[4,7],both:4,bound:[4,5],broadcast:5,btw:6,buff:5,buffer:[5,6,7],buffer_len:5,buffer_s:6,build:[4,6,7],build_processing_chain:[0,2,3],build_tabl:6,built:7,bulk:0,c:4,caen:6,caendt5725:6,caendt5730:6,caendt57xx:6,cal_db:3,cal_par:4,cal_peak:4,cal_pk:4,cal_slop:4,cal_typ:4,calcul:4,caldwel:6,calibr:[0,2,3,6],calibraiton:4,calibrate_tl208:4,call:[3,4,5,6,7],can:[1,3,4,5,6,7],cannot:[4,5],canon:7,card:6,care:[3,4],ccc:6,center:4,centroid:4,certain:6,ch:6,ch_group:[0,2,3],ch_groups_dict:6,ch_list:6,ch_to_tbl:6,chain:5,chan:6,chan_info:4,chang:[4,7],channel:6,channelconfig:6,channelindex:6,check_indic:5,chi2:4,chi2r:4,chisq:4,chisquar:4,choic:4,choos:[3,4],class_kei:6,class_nam:6,clear:7,clint:3,clk:5,clock:5,clock_unit:5,clone:1,cmd:3,co:[3,4],code:[6,7],coeff:4,col:7,col_dict:7,collect:6,column:[4,7],com:[1,3],combin:6,command:[3,7],comment:3,common:4,compar:4,compass:6,compassdaq:[0,2,3],compat:[5,6],compon:4,compress:6,compton:4,comput:[3,4],concaten:7,config:[4,5,6],configur:6,conjunct:7,consist:4,constant:[4,5],constitut:3,constraint:7,construct:5,constructor:[6,7],contain:[3,4,5,6,7],contaten:7,content:[0,2],contigu:7,contiguosli:7,contingu:6,contribut:6,conveni:[3,4],convent:[4,5],convers:4,convert:[0,3,4,5,6],copi:[5,7],correct:5,correctli:5,correspond:[4,5,6],could:[6,7],count:4,cov:[3,4],cov_list:4,cov_matrix:4,covari:[3,4],crate:6,creat:[4,5,6],create_datafram:6,create_dummy_ch_group:6,criteria:7,cross:4,crystal_ball_funct:4,crystalbal:4,csv:4,cumul:7,cumulative_length:[6,7],current:[3,4,6],curvatur:4,curve_fit:[3,4],cut:4,cut_sigma:4,d:[6,7],daq:[4,6],daq_fil:6,daq_filenam:6,daq_to_raw:[0,2,3],data:[0,3,4,5,6,7],data_buff:6,data_clean:[0,2,3],data_id:6,data_pk:4,databas:[4,5],datadecod:6,datadescript:6,datafram:[4,7],datagroup:[0,2,3],dataid:6,datatak:6,datatyp:7,datatype_nam:7,dataype_nam:7,david:4,db:5,db_dict:[4,5],dbl:4,de:4,debas:3,debug:[5,7],declar:3,decod:6,decode_ev:6,decode_packet:6,decoded_valu:6,decoder_nam:6,decodernam:6,decompress:6,deduc:5,deduct:5,defaultdict:7,defin:[4,5,6],definit:4,deg:4,degre:4,delta:[3,4],denomin:4,deprec:4,depth:4,deriv:4,describ:6,desir:4,detect:4,detected_peak_energi:4,detected_peak_loc:4,detected_peaks_kev:4,detected_peaks_loc:4,detectorname1:6,detectorname2:6,determin:[4,6,7],develop:1,df:[4,6,7],dfwfm:4,dfwhm:4,diagram:6,dict:[3,4,5,6,7],dictionari:[5,6,7],diffenc:4,differ:[4,5,6,7],digit:[0,5,6],dim:[5,7],dimens:[5,7],dimension:[5,7],dir:4,direct:4,directli:[4,6,7],directori:[3,4,6],discuss:3,disk:4,distribut:[3,4],divid:[4,5],dmu:4,dmx:4,do_warn:7,doc:4,docstr:3,document:[3,6],doe:[3,4],doesn:[3,5,6],dof:4,domain:4,domin:4,don:[3,4],done:5,doubl:7,double_gauss:4,down:[4,5],draw:3,ds:3,dsp:[0,2,3],dsp_config:5,dsp_optim:[0,2,3],dtype:[5,7],dummi:6,dure:6,dx:4,e:[1,3,4,5,6,7],e_cal:4,e_kev:4,e_scal:4,e_scale_par:4,e_unc:4,each:[3,4,5,6,7],easi:[3,6],easier:[3,5],easili:4,edg:4,edit:1,effici:5,either:[4,5,7],el_typ:7,element:[4,7],els:5,elsewher:6,empti:6,en:4,enabl:5,end:[4,5],energi:[4,6],energy_seri:4,energyseri:4,enough:[4,5],entir:5,entri:[4,5,6,7],equal:[5,7],equival:4,error:[4,5],es_kev:4,essenti:7,estim:4,etc:[3,5,6],etol:4,etol_kev:4,euc_max:4,euc_min:4,evalu:4,even:4,event:[0,4,6],event_data:6,event_data_byt:6,event_numb:6,ever:3,everi:6,everyth:5,exampl:[4,6,7],except:[5,6],exclusevli:4,execut:5,execute_block:5,exist:5,exp:[1,4,6],expand:[6,7],expand_ch_group:6,expans:4,expect:4,experi:[4,5],explain:3,explicitli:6,expos:3,expr:5,express:5,extend:6,extern:5,extra:4,extract:[3,4,5],extrema:4,extremum:4,f:4,f_db:4,f_in:6,f_likelihood:4,f_list:7,f_out:6,factor:4,factori:5,fadc:6,fadcindex:6,fail:5,fall:4,fals:[3,4,5,6,7],fanci:7,farther:4,fast:3,fav:3,fave:4,fc:6,fcdaq:[0,2,3],fcio:6,fcioconfig:6,fcioevent:6,fcutil:6,fed:5,fenc:4,fetch:5,few:7,field:[5,6,7],field_mask:7,figur:[3,4,5],figure_of_merit:5,file:[0,3,4,5,6,7],file_binari:6,filedb:4,filehead:6,fileid:4,filenam:[4,6],filename_g:6,filename_muvt:6,fill:6,find:[4,5,7],find_bin:4,find_pulser_properti:4,fine:4,first:[3,4,5,6,7],fit:[3,4],fit_bin:4,fit_hist:4,fit_simple_sc:3,fit_slop:4,fit_unbin:4,fix:[5,7],fixedsizearrai:[0,2,3],flag:[1,4],flashcam:6,flashcameventdecod:6,flashcamstatusdecod:6,flat:3,flatten:7,flattened_data:[6,7],flip:6,flip_data_id:6,fname:4,folder:4,follow:[4,5,6],fom:5,fom_funct:5,fom_kwarg:5,form:[3,4,5],form_datatyp:7,format:[0,3,4,5,6,7],format_data:6,found:[4,5,6],fraction:4,fration:4,free:4,freedom:4,friggin:5,from:[3,4,5,6,7],from_byt:6,full:4,func:[3,4,5],further:0,furthermor:5,futur:4,fwfm:4,fwhm:4,g:[1,4,5,6,7],gain:6,gamma:4,garbag:6,garbage_length:6,gauss:4,gauss_bas:4,gauss_bkg:4,gauss_cdf:4,gauss_int:4,gauss_lin:4,gauss_mode_max:4,gauss_mode_width_max:4,gauss_step:4,gauss_tail:4,gauss_tail_approx:4,gauss_tail_exact:4,gaussian:4,gaussian_cut:4,ged:6,gener:[0,3,4],get:[4,5,6,7],get_bin_cent:4,get_bin_estim:4,get_bin_width:4,get_buff:7,get_calibration_energi:4,get_card:6,get_ccc:6,get_channel:6,get_crat:6,get_data:5,get_data_id:6,get_datafram:7,get_dataset_from_cmdlin:3,get_decoded_valu:6,get_detectors_for_preamp:6,get_ev:6,get_event_s:6,get_file_config_struct:6,get_formatted_stat:3,get_fwfm:4,get_fwhm:4,get_fwhm_func:4,get_gaussian_guess:4,get_grid_point:5,get_hist:4,get_hpge_e_peak_bound:4,get_hpge_e_peak_par_guess:4,get_i_local_extrema:4,get_i_local_maxima:4,get_i_local_minima:4,get_id_to_decoder_name_dict:6,get_input_buff:5,get_lh5_col:4,get_lh5_element_typ:7,get_list_of:6,get_most_prominent_peak:4,get_mu_func:4,get_n_dimens:5,get_n_grid_point:5,get_n_points_of_dim:5,get_name_onli:5,get_names_onli:5,get_next_packet:6,get_object_info:6,get_output_buff:5,get_par_meshgrid:5,get_par_nam:3,get_run_numb:6,get_shap:5,get_vari:5,get_zero_indic:5,getbound:4,gimme_fil:7,gimme_group:7,git:[0,1,2],github:1,give:[4,6],given:[3,4,5,6,7],global:6,go:4,goe:6,gof:4,gof_method:4,good:4,goodness_of_fit:4,got_peak_energi:4,got_peak_loc:4,gotnrofbyt:6,grab:6,greater:4,gretina4m:6,grid:5,grid_valu:5,group:[3,6,7],group_nam:6,group_path:6,group_path_templ:6,grp_attr:7,grp_path_templ:6,guarante:[4,7],guassian:4,guess:4,guess_kev:4,gufunc:5,guvector:5,h5py:7,ha:[4,5,6,7],handl:[3,5,6,7],have:[1,3,4,5,6],hdf5:[0,6],header:6,header_dict:6,height:4,held:5,help:[3,4,6],here:[3,4,5,6],high:[0,4],hist:4,histogram:[0,2,3],hold:[4,7],hopefulli:4,how:[4,6],howev:4,hpge:4,hpge_e_calibr:4,hpge_find_e_peak:4,hpge_fit_e_cal_func:4,hpge_fit_e_peak:4,hpge_fit_e_peak_top:4,hpge_fit_e_scal:4,hpge_get_e_peak:4,hstep:4,htail:4,html:[3,4],http:[1,3,4,6],human:3,hv:6,i1:5,i2:5,i:[3,4,5,6,7],i_dim:5,i_match:4,i_par:5,i_vec:7,id:6,ideal:5,ident:7,identif:7,identifi:[4,6],idx:7,idx_list:7,ignor:4,iii:5,il:[3,4],imax:4,imin:4,immedi:7,implement:[4,5,6],importantli:6,includ:[3,4,5,6,7],incorpor:4,index:[4,5,6,7],indic:[4,5,6],indico:6,individu:6,inf:[4,6],inflat:4,inflate_error:4,info:[5,6,7],inform:[5,6,7],init:3,init_arg:5,init_obj:6,initi:[4,5,6],initialize_lh5_t:6,input:[3,4,5,6],input_config:6,insert:7,insid:6,instal:0,instanti:7,instead:[4,5],instruct:1,integ:[4,6],integr:4,intent:6,intercept:3,interfac:5,intern:[5,7],interpol:4,interpret:[4,7],invers:4,invert:6,io:[0,2,3],io_bas:[0,2,3],ipython:3,is_ful:7,is_multisampl:6,isclos:4,iseg:6,iseghvdecod:6,item:7,iter:5,iterate_indic:5,ith:4,its:[3,4,5,6,7],jason:4,json:[4,5,6],jump:4,jupyterhub:1,just:[3,4,6,7],k:[4,6],keep:7,keep_open:7,kei:[3,4,5,6,7],kept:4,kev:4,keyword:5,known:4,kwarg:[3,4,5,6],l:6,label:[3,6],largest:4,laru:6,last:[4,6],law:4,least:[3,4],leav:4,left:4,legend:[0,1,3,4,5,6],len:[4,5,7],length:[4,5,6,7],less:4,level:[0,4,5],lh5:[0,2,3,4,5],lh5_dir:4,lh5_dir_setup:4,lh5_file:7,lh5_group:7,lh5_in:5,lh5_out:5,lh5_store:6,lh5_tabl:6,lh5_user:4,lh5_util:[0,2,3],librari:3,like:[3,4,6,7],likelihood:4,line:[3,4],linear:[3,4],linear_fit_by_sum:3,link:[1,5],list:[3,4,5,6,7],literatur:4,llama:6,llama_3316:6,llamadaq:[0,2,3],llamastruck3316:6,ln:4,load:[3,4,7],load_df:[4,7],load_kei:4,load_nda:7,loc:7,local:[1,4],locat:[4,5,7],log:[4,6],lone:4,look:[4,5,6,7],lookup:4,loop:[5,7],low:4,lower:4,ls:7,m1:4,m2:4,m:[3,4],made:[4,5,6,7],magic:6,mai:[4,5],main:[4,6],mainli:4,maintain:7,majorana:6,make:[3,4,6],manag:7,mani:[3,4,5],map:6,mario:6,match:[4,6],match_peak:4,matlab:[3,4],matric:4,matrix:[4,5],max:[4,7],max_n_rows_per_packet:6,max_num_peak:4,maxima:4,maximum:[4,7],maxtab:3,mayb:6,me:4,mean:[3,4],measur:4,member:6,memori:[4,5,7],merit:5,meshgrid:5,meta:6,metadata:[0,2,3,5,6],method:[4,5,6],mg:5,mgdodecod:[0,2,3],might:3,min:4,min_method:4,minim:4,minima:4,mintab:3,miss:3,mjd_data_format:6,mjdpreamp:6,mjdpreampdecod:6,mode:[1,4,7],mode_guess:4,model:3,model_nam:6,moder:4,modul:[0,2],module_or_subpackag:3,module_thats_next_alphabet:3,monoton:4,more:[3,7],most:[4,6],mpl:3,mth:5,mu1:4,mu2:4,mu3:4,mu:4,mu_var:4,multi:5,multipl:[4,5],multipli:[5,6],must:[4,5,6,7],muvt:6,mx:[3,4],n:[4,5,7],n_bin:4,n_max:6,n_row:7,n_rows_read:7,n_sigma:4,n_slope:4,n_to_fit:4,name:[3,4,5,6,7],narg:3,natur:5,nda:[6,7],nda_to_vect:6,ndarrai:[3,5,6,7],ndig:3,nearest:5,necessari:5,need:[4,5,6,7],neg:4,neg_log_lik:4,neg_poisson_log_lik:4,nersc:1,nest:[3,5],never:7,new_siz:7,next:6,neyman:4,nfile:4,nice:[3,6],non:[4,5,7],none:[3,4,5,6,7],nonlinear:4,normal:[1,4],note:[3,4,5,6,7],noth:5,now:7,np:[4,7],npx:3,nr:6,ns:6,ntupl:5,num:3,numba:5,number:[3,4,5,6,7],numer:[4,5,7],numpi:[3,4,5,7],o:[4,5,6,7],oad:7,obei:7,obj:7,obj_buf:7,obj_buf_start:7,obj_dict:7,object:[4,5,6,7],object_info:6,obtain:4,off:4,offset:[4,5],often:[3,4],okai:4,old:4,omit:1,onc:[5,6],one:[3,4,5,6,7],ones:4,onli:[3,4,5,7],onto:7,open:6,open_orca:6,oper:5,oppos:5,opposit:7,opt:5,optim:[0,3,4,5,7],option:[3,4,5,6,7],orca:6,orca_class_nam:6,orca_digit:[0,2,3],orca_filenam:6,orcadaq:[0,2,3],orcadecod:6,orcagretina4m:6,orcastruck3302:6,order:[4,5],org:[4,6],orrunmodel:6,orsis3302decoderforenergi:6,other:[3,4,5],otherwis:[4,5,6],out:[4,5,6,7],out_dir:6,out_fil:6,out_file_templ:6,output:[0,3,5,6,7],output_dir:6,outsid:4,over:[4,5,7],overflow:4,overload:6,overrid:4,overwhelmingli:4,overwrit:6,own:[3,6],p12345a:6,p12346b:6,p:4,packag:2,packet:6,packet_id:6,packet_size_guess:6,pad:3,page:0,pair:4,panda:[6,7],pandas:7,par:[3,4,5],par_data:7,par_list:7,param:[3,4],paramat:4,paramet:[0,3,4,5,6,7],pargrid:5,pargriddimens:5,pars:[4,5,6,7],pars_guess:4,pars_list:4,parse_channelconfig:6,parse_datatyp:7,parse_filehead:6,parse_head:6,parser:6,part:6,partit:4,pass:[4,5,6,7],path:[4,6,7],pattern:[4,5],pdf:6,pdfviewer:4,peak:4,peak_energi:4,peak_fit:[0,2,3],peak_loc:4,peakdet:[3,4],peaks_kev:4,peaks_unc:4,pearson:4,per:6,percentag:4,perform:[0,4,5],pgf:4,pgh:4,ph:4,php:4,physic:0,picket:4,pip:1,pk:4,pk_binw:4,pk_cal_cov:4,pk_cal_par:4,pk_cov:4,pk_par:4,pk_rang:4,place:4,plan:1,plate:6,platform:7,plot:[3,4],plot_func:3,plot_hist:4,plotaxi:4,plotfigur:4,plu:4,point:[3,4,5],pointer:6,poisson:4,poisson_gof:4,poissonl:4,poke:3,pol:4,poli:4,poller:[0,2,3],poly_match:4,polyfit:[3,4],polynomi:4,posit:4,power:4,pre:7,preamp_id:6,prefer:3,prereq:5,present:4,preserv:4,pretti:4,previou:7,print:[4,5,7],print_data:5,print_fit_result:3,printmessag:6,prior:5,probabl:4,problemat:6,proc_chain:5,procchain:5,process:[0,5,6],process_compass:6,process_flashcam:6,process_llama_3316:6,process_orca:6,processingchain:[0,2,3],processor:[0,2,3],produc:[5,6],program:6,progress:[0,3],promin:4,proper:[4,7],properli:6,provid:[4,5,6,7],pt_cal_cov:4,pt_cal_par:4,pt_cov:4,pt_par:4,publicli:3,pul:6,pulse_shape_analysi:[0,2,3],pulser:4,push_row:7,put:[3,4,5,6],put_in_garbag:6,py:[3,6],pygamaland:3,python2:6,python:[0,3,4],quick:6,quickli:3,quot:3,r:3,radford:4,radford_fwhm:4,radford_parameter_gradi:4,radford_peak:4,radford_peak_wrap:4,radford_peakshape_deriv:4,random:4,rang:[3,4,5,6],range_kev:4,range_slic:4,rate:4,rather:[3,6],ratio:[4,5],raw:6,raw_fil:6,raw_file_pattern:6,raw_filenam:6,raw_to_dsp:[0,2,3],rb:6,read:[5,6,7],read_n_row:7,read_next_ev:6,read_object:7,readabl:3,readm:6,readmetadata:6,readout:6,reason:[4,5],recip:5,record:6,reddit:3,redund:7,refer:[0,4],refin:4,regener:4,region:4,regular:4,rel:4,relationship:4,remov:7,replac:4,represent:7,req:5,request:7,requestednrofbyt:6,requir:[4,5,6],resiz:7,result:[3,4,6],rewrit:6,right:[1,3,4,5,6],risetim:5,root:[3,4],rough:4,roughli:4,round:5,routin:[0,4],row:[5,7],rtol:4,run:[1,3,5,6],run_db:3,run_grid:5,run_grid_multiprocess_parallel:5,run_grid_point:5,run_multi_grid:5,run_one_dsp:5,s1:4,s2:4,s:[3,4,5,6,7],safedict:3,sai:4,same:[4,5,6,7],sampl:6,sample_period:6,sample_r:6,satisfi:4,save:[4,6],save_df:4,save_kei:4,scalar:[0,2,3,5],scale:[3,4],scale_var:3,scan:4,scan_daq_dir:4,scimath:5,scipi:[3,4],script:[3,4],search:[0,4],second:[4,5],see:[4,5,6,7],select:[0,5,6,7],self:[4,5,6,7],send:[4,5,7],sens:[3,4],sent:4,separ:[4,5],sequenc:[4,5],seri:[0,5],set:[1,4,5,6,7],set_config:4,set_dsp_par:5,set_file_config:6,set_header_dict:6,set_object_info:6,set_output:6,set_plot_styl:3,set_vector:7,sh:3,shape:[4,5,7],shape_guess:7,shell:3,shift:4,shorthand:6,should:[3,4,5,6,7],show_stat:4,sig_in:6,sig_len_in:6,sig_out:6,sigma1:4,sigma2:4,sigma3:4,sigma:[3,4],signal:[0,5],signatur:5,signific:3,simd:5,similar:4,simpl:[3,4],simplest:4,singl:3,sis3316:6,situat:4,size:[3,4,5,7],sizeof_fmt:3,slice:[4,5,7],slide:6,slope:3,slow:5,smaller:7,so:[3,4,5,6,7],some:[4,6,7],somehow:6,someth:[3,4],sophist:7,sort:4,sourc:[3,4,5,6,7],space:[4,5],spars:5,special:[4,7],specif:[3,4,6,7],specifi:[3,4,5,6,7],spectrum:4,sphinx:3,spike:4,split:7,spm:6,sqrt:4,squar:[3,4],stabl:4,stackoverflow:3,start:[5,6,7],start_guess:4,start_row:7,stat:4,state:7,stats_hloc:4,stats_vloc:4,statu:6,step:4,store:[0,2,3,4,5,6],str:[4,5,6,7],stream:6,string:[3,4,5,6,7],struck:6,struct:[0,2,3,6],structur:6,style:[3,7],sub:[5,6],subclass:6,subfield:7,submodul:[0,2],subpackag:[0,2],subrang:5,subrun:6,subtabl:7,successfulli:7,suffix:3,sum:[4,7],summari:6,super1:6,super2:6,super_nam:6,suppli:[4,6],support:[4,7],sure:4,syntax:5,sysn:6,system:[0,1,6],t0_file:6,t:[3,4,5,6],tabl:[2,3,5,6],tag:4,tag_puls:4,tail:4,take:[3,4,6],taker:6,tallest:4,task:4,tau:4,taylor:4,taylor_mode_max:4,tb_data:5,tb_out:5,tcut:3,technic:4,temp:6,templat:[4,6],term:4,test:4,than:[3,4,6,7],thank:5,thei:[4,5],them:[4,5,6],thi:[1,3,4,5,6,7],thing:[3,4,5,6],thing_i_w:3,thingiw:3,think:3,thispackag:3,those:[0,3,4,6,7],though:3,three:4,threshold:4,through:5,thu:4,tier:6,time:[0,3,4,5,6],tinydb:4,titl:3,todo:[4,6,7],toler:4,tom:6,top:4,total:6,track:7,transform:[3,5],trap:5,trap_max:4,travers:[4,5],tree:3,tree_draw:3,trickier:4,tripl:3,truncat:7,ttree:3,tune:0,tupl:[3,4,5,7],turn:7,two:[3,4,6,7],type:[3,4,5,6,7],typic:[4,5],ufunc:5,uint32:6,unambigu:4,unari:5,unbin:4,uncal_is_int:4,uncalibr:[4,6],uncertainti:4,underflow:4,underli:[4,7],uniform:[4,6],uninstal:1,unique_kei:4,unit:[0,2,3,4,7],unix:7,unknown:5,unless:4,unnecessari:4,until:[4,7],up:[1,4,5,6,7],updat:[3,6,7],update_datatyp:7,update_progress:3,upon:5,upper:4,us:[3,4,5,6,7],use_obj_s:7,user:[1,3,4,6],user_dir:4,usual:4,util:[0,2,4],v:[3,6],valid:[6,7],valu:[3,4,5,6,7],value_str:5,var_zero:4,vari:[4,5],variabl:[3,4,5,6,7],varianc:[3,4],varnam:5,ve:4,vect_to_nda:6,vector:[3,5,7],vectorofvector:[0,2,3],verbos:[4,5,6,7],veri:4,version:[3,4,6],via:[3,4],vicin:4,voltag:6,w:4,wa:[4,6],wai:[4,7],want:[3,4,7],warn:5,waveform:[5,7],waveformbrows:[0,2,3],we:[1,3,4,5,6,7],weight:[3,4],well:[4,6],were:4,wfcompress:[0,2,3],what:3,whats_your_opinion_on_what_to_include_in_init_pi:3,when:[3,4,5,6,7],where:[4,5,6,7],wherea:4,whether:4,which:[3,4,5,6,7],whose:[4,5,6],width:[4,5],wiki:4,wikipedia:4,wildcard:7,window:4,within:[4,7],without:[4,5],word:[5,6],work:[0,4,6],would:[3,4,5],wrap:4,wrapper:4,write:[4,5,6,7],write_object:7,write_out_garbag:6,write_pretti:4,written:[6,7],wt:4,wwidth:4,www:3,x:[3,4,6],x_hi:4,x_high:4,x_lo:4,x_max:4,x_min:4,xhi:4,xlo:4,xpb:4,xtalbal:4,xtalball_cut:4,xx:4,xxx:6,y:[3,6],yet:4,you:[1,3,4,5,6,7],your:[3,4,6],yourself:3,yy:4,z:6,zero:4},titles:["Welcome to pygama's documentation!","Installing pygama","pygama","pygama package","pygama.analysis package","pygama.dsp package","pygama.io package","pygama.lh5 package"],titleterms:{analysi:4,arrai:7,arrayofequalsizedarrai:7,build_processing_chain:5,calibr:4,ch_group:6,compassdaq:6,content:[3,4,5,6,7],daq_to_raw:6,data_clean:4,datagroup:4,document:0,dsp:5,dsp_optim:5,fcdaq:6,fixedsizearrai:7,git:3,histogram:4,index:0,indic:0,instal:1,io:6,io_bas:6,lh5:[6,7],lh5_util:7,llamadaq:6,metadata:4,mgdodecod:6,modul:[3,4,5,6,7],orca_digit:6,orcadaq:6,packag:[0,3,4,5,6,7],peak_fit:4,poller:6,processingchain:5,processor:5,pulse_shape_analysi:4,pygama:[0,1,2,3,4,5,6,7],raw_to_dsp:6,s:0,scalar:7,store:7,struct:7,submodul:[3,4,5,6,7],subpackag:3,tabl:[0,7],unit:5,util:3,vectorofvector:7,waveformbrows:5,welcom:0,wfcompress:6}})