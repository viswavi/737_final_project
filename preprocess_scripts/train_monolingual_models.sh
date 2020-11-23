./preprocess_scripts/make-ted-bilingual.sh aze data_monoaugment_for_M2O
./preprocess_scripts/make-ted-bilingual.sh aze data_monoaugment_for_O2M
./preprocess_scripts/make-ted-bilingual.sh bel data_monoaugment_for_M2O
./preprocess_scripts/make-ted-bilingual.sh bel data_monoaugment_for_O2M

./preprocess_scripts/make-ted-multilingual.sh aze tur data_monoaugment_for_O2M
./preprocess_scripts/make-ted-multilingual.sh aze tur data_monoaugment_for_M2O
./preprocess_scripts/make-ted-multilingual.sh bel rus data_monoaugment_for_O2M
./preprocess_scripts/make-ted-multilingual.sh bel rus data_monoaugment_for_M2O

./job_scripts/monolingual_trainer.sh eng bel data_monoaugment_for_O2M
./job_scripts/monolingual_trainer.sh bel eng data_monoaugment_for_M2O
./job_scripts/monolingual_trainer.sh eng aze data_monoaugment_for_O2M
./job_scripts/monolingual_trainer.sh aze eng data_monoaugment_for_M2O

./job_scripts/multilingual_trainer.sh eng rus bel data_monoaugment_for_O2M
./job_scripts/multilingual_trainer.sh bel rus eng data_monoaugment_for_M2O
./job_scripts/multilingual_trainer.sh eng tur aze data_monoaugment_for_O2M
./job_scripts/multilingual_trainer.sh aze tur eng data_monoaugment_for_M2O
