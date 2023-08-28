<!-- This is the scheduler and data switches of Secure MLaaS -->
# Scheduler
1. config: model-slave-address. Initialize the table.
    1. Mapping ip addresses to slaves.
    2. Mapping models to slaves.
2. FIFO queue for each slave, storing the apply of data switch.
3. Look-up table for each data switch. When a data switch asks for a model, scheduler initiates a queue for it, containing the ip addresses of corresponding slaves. 
# Working process
0. Create secure channel between user and data switch. 
1. Recieve data and analyze it(data switch).
2. Validate data integrity and form(data switch).
3. Data switches query scheduler for a idle slave. 
4. Scheduler sets a FIFO queue of data switch for each slave. The queue keeps the address of data switch. 