SRC=Allen/src
LOG=dataset.csv
OPTQ=-q'/^EMeter_/>=20,/\.BatteryLevel$$/>=20'
FAULTS=door_failure_1week bed_failure toilet_failure
RULES=sleep_quiet toilet outing

%.pm: %.aln
	$(SRC)/allenc $<

work/%.py: %.py
	cp $< $@

detect: work rules days scripts

work:
	mkdir $@

scripts: $(RULES:%=work/visualize_%.py)

out/dataset: log-analyses.pm
	cat $ dataset.csv | \
		$(SRC)/allen $(OPTQ) log-analyses.pm | sort >$@

work/rule-%.csv: out/dataset
	grep ";$(@:work/rule-%.csv=%);" $< >$@

work/days-%.csv: work/rule-%.csv
	./dayspermonth.pl -b 2017-01-01T00:00:00 -e 2017-12-31T02:00:00 $< >$@


rules: $(FAULTS:%=work/rule-%.csv) $(RULES:%=work/rule-%.csv)

days: $(FAULTS:%=work/days-%.csv) 

server:
	docker run -p 8888:8888 -e NB_USER=`id -n -u` -v `pwd`/work:/home/jovyan/work:rw jupyter/scipy-notebook:2c80cf3537ca

detect-indock:
	docker run -w=/home -v `pwd`:/home:rw perl:5.30 make detect

clean:
	rm -f work/* out/* log-analyses.pm