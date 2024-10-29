from experiment_setup import PhaseRetrieval_Experiment


pre = 'params/mirror_params_phase'
#pre = 'params/sphere_phase'

for conf_name in ['N0', 'N1', 'N2', 'N3']:
    num_runs = 20
    success = 0
    conf = PhaseRetrieval_Experiment(pre + conf_name + '.yaml')
    #%%
    for run in range(num_runs):
        xin = conf.init_x()
        dyn = conf.dyn_cls(conf.get_objective(), x = xin, **conf.dyn_kwargs)
        
        #%%
        dyn.optimize(print_int=100, sched=conf.get_scheduler())
        
        ev = conf.eval_run(dyn)
        success += ev['success']
        print('Run M = ' + str(conf.config.problem.M) + ', num success = ' + str(success) + ' dist= '
              + str(ev['consensus_diff'][-1]))
    #%%
    print(30*'<>-<>')
    print('Run M = ' + str(conf.config.problem.M) + ' noise = ' + str(conf.config.problem.sigma_noise))
    print('Success Rate: ' + str(success/num_runs))