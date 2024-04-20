# Advantage Alignment 

## Update log: 04/20/2024 by Tianyu

1. include `tournament.yaml` and `tournament.py`. Usage: `python tournament.py`.
The definition of the `aa` and `ppo` agent network information is [here](https://github.com/jduquevan/advantage-alignment/blob/master/configs/tournament.yaml#L28). All the network-based agent must share the same architecture. The tornament between agents with different architectures is not supported yet.   
2. include agent load, save, eval, train. Check [here](https://github.com/jduquevan/advantage-alignment/blob/master/src/agents/agents.py#L17)
Usage:
```python
agent.to(cfg.device)
agent.save('agent.pth') # save all the nn.Module state_dict in the agent
agent.load('agent.pth') # load all the nn.Module state_dict in the agent
agent.eval()
agent.train()
```
3. implement `detach_and_move_to_cpu` for the `trajecotry` class to be able to save to json.

