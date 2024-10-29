from experiment_setup import PhaseRetrieval_Experiment



pre = 'params/mirror_params_phase'
for conf_name in [pre +'500.yaml']:
    num_runs = 2
    success = 0
    conf = PhaseRetrieval_Experiment(conf_name)
    #%%
    for run in range(num_runs):
        xin = conf.init_x()
        dyn = conf.dyn_cls(conf.get_objective(), x = xin, **conf.dyn_kwargs)
        
        #%%
        dyn.optimize(print_int=100, sched=conf.get_scheduler())
        
        ev = conf.eval_run(dyn)
        success += ev['success']
    #%%
    print(30*'<>-<>')
    print('Run M= ' + str(conf.config.problem.M))
    print('Success Rate: ' + str(success/num_runs))