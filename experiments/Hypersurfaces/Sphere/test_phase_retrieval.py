from experiment_setup import PhaseRetrieval_Experiment

num_runs = 2
success = 0
conf = PhaseRetrieval_Experiment('params/mirror_params_phase.yaml')

for M in [500, 1000]:
    #%%
    for run in range(num_runs):
        conf.config.problem.M = M
        xin = conf.init_x()
        dyn = conf.dyn_cls(conf.get_objective(), x = xin, **conf.dyn_kwargs)
        
        #%%
        dyn.optimize(print_int=100, sched=conf.get_scheduler())
        
        ev = conf.eval_run(dyn)
        success += ev['success']
    #%%
    print(30*'<>-<>')
    print('Run M= ' + str(M))
    print('Success Rate: ' + str(success/num_runs))