@echo off
python test_agent.py --env VariableWindLevel0-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"
python test_agent.py --env VariableWindLevel1-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"
python test_agent.py --env VariableWindLevel2-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"
python test_agent.py --env VariableWindLevel3-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"
python test_agent.py --env VariableWindLevel4-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"
python test_agent.py --env VariableWindLevel5-v17 --psf --num_episodes 100 --time 600 --agent "logs/VariableWindLevel2-v17/1618928488ppo/agents/last_model_10000000.zip"