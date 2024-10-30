from experiment_setup import PhaseRetrieval_Experiment
from phase_retrieval import WirtingerFlow

run_Wirtinger = True
pre = 'params/mirror_params_phase'
pre = 'params/sphere_phase'

for conf_name in ['1500']:
    num_runs = 5
    Wirtinger_success = 0
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
        print('Run: ' +str(run) + ', M = ' + str(conf.config.problem.M) + 
              ' noise = ' + str(conf.config.problem.sigma_noise))
        print('num success = ' + str(success) + ' dist= ' + str(ev['consensus_diff'][-1]))
        
        #%%
        if run_Wirtinger:
            Wirtinger_success += conf.eval_Wirtinger_Flow()['success']
            print('Wirtinger Success: ' + str(Wirtinger_success))
        

    #%%
    print(30*'<>-<>')
    print('Run M = ' + str(conf.config.problem.M) + ' noise = ' + str(conf.config.problem.sigma_noise))
    print('Success Rate: ' + str(success/num_runs))
    if run_Wirtinger:
        print('Wirtinger Success: ' + str(Wirtinger_success/num_runs))
    print(30*'<>-<>')
