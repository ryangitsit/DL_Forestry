CVEPOCHS = 50
EPOCHS = 101

crossvalidation_experiments:
	for LR in 0.001 0.01 0.05 0.1 ; do \
		for MOM in 0 0.1 0.5 1.0 ; do \
			python main.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM ;\
			python main.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm ;\
			python main.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm --activation=elu ;\
			python main.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --activation=elu ;\
		done ;\
	done

crossvalidation_plots:
	for LR in 0.001 0.01 0.05 0.1 ; do \
		for MOM in 0 0.1 0.5 1.0 ; do \
			python generate_plot.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM ;\
			python generate_plot.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm ;\
			python generate_plot.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --optimizer=sgdm --activation=elu ;\
			python generate_plot.py --epochs=$(CVEPOCHS) --crossvalidation=True --learningrate=$$LR --momentum=$$MOM --activation=elu ;\
		done ;\
	done



final_experiments:
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=sgdm --activation=elu --augmentation=True ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=rms --activation=elu --augmentation=True ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.05 --momentum=0.5 --optimizer=rms --activation=relu --augmentation=True ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.1 --momentum=0.0 --optimizer=sgdm --activation=relu --augmentation=True ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=sgdm --activation=elu ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=rms --activation=elu ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.05 --momentum=0.5 --optimizer=rms --activation=relu ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.1 --momentum=0.0 --optimizer=sgdm --activation=relu ;\

final_plots:
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=sgdm --activation=elu --augmentation=True ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=rms --activation=elu --augmentation=True ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.05 --momentum=0.5 --optimizer=rms --activation=relu --augmentation=True ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.1 --momentum=0.0 --optimizer=sgdm --activation=relu --augmentation=True ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=sgdm --activation=elu ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --optimizer=rms --activation=elu ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.05 --momentum=0.5 --optimizer=rms --activation=relu ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.1 --momentum=0.0 --optimizer=sgdm --activation=relu ;\

experiment:
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --augmentation=True ;\
			python main.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 ;\

experiment_plot:
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 --augmentation=True ;\
			python generate_plot.py --epochs=$(EPOCHS)  --learningrate=0.01 --momentum=0.5 ;\
	
