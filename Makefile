SRC=Allen/src
LOG=dataset.csv
OPTQ=-q'/^EMeter_/>=20,/\.BatteryLevel$$/>=20'
FAULTS=door_failure_1week bed_failure toilet_failure platform_failure_1day
RULES=sleep_quiet toilet outing breakfast_7_9 lunch_12_14 dinner_18_21

# Automatically determine the next available participant number
NEXT_PARTICIPANT=$(shell expr `ls -d work/participant_* 2>/dev/null | sed 's|.*/participant_||' | grep -E '^[0-9]+$$' | sort -n | tail -n1 || echo 0` + 1)
PARTICIPANT_DIR=work/participant_$(NEXT_PARTICIPANT)

# Compile .aln into .pm
%.pm: %.aln
	$(SRC)/allenc $<

# Copy .py scripts to work dir
work/%.py: %.py
	cp $< $@

CURRENT_PARTICIPANT=$(shell expr $(NEXT_PARTICIPANT) - 1) #quand on créer le dossier participant_2 le next_participant passe à 3 automatiquement
detect: work rules days copy_to_participant
	sleep 1
	python3 work/process_failure.py $(CURRENT_PARTICIPANT)

# Create working directory
work:
	mkdir -p $@

# Rule targets
rules: work/rules $(FAULTS:%=work/rules/rule-%.csv) $(RULES:%=work/rules/rule-%.csv)

# Days targets
days: work/sensors_failure_days $(FAULTS:%=work/sensors_failure_days/%_days.csv)

# Create subdirectories
work/rules:
	mkdir -p work/rules

work/sensors_failure_days:
	mkdir -p work/sensors_failure_days

# Generate dataset file from log-analyses
out/dataset: log-analyses.pm
	cat $(LOG) | \
		$(SRC)/allen $(OPTQ) log-analyses.pm | sort >$@

# Generate rule CSVs
work/rules/rule-%.csv: out/dataset
	grep ";$*;" $< >$@

# Generate days CSVs
work/sensors_failure_days/%_days.csv: work/rules/rule-%.csv
	./dayspermonth.pl -b 2017-01-01T00:00:00 -e 2017-12-31T02:00:00 $< >$@

# Copy rules and days to participant folder
copy_to_participant:
	mkdir -p $(PARTICIPANT_DIR)
	mv work/rules $(PARTICIPANT_DIR)/
	mv work/sensors_failure_days $(PARTICIPANT_DIR)/
	mkdir -p $(PARTICIPANT_DIR)/new_processed_csv

# Cleanup
clean:
	rm -rf work/* out/* log-analyses.pm
