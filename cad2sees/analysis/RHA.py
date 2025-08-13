import openseespy.opensees as ops
import numpy as np


def _doSingleRHA(Duration, dt,
                 TestTyp='NormDispIncr',
                 tolInit=1e-5,
                 iterInit=20,
                 algorithmTyp='KrylovNewton',
                 ModeNum=0):
    ops.test(TestTyp, tolInit, iterInit)
    ops.algorithm(algorithmTyp)
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    THAFlag = 0
    CurTime = ops.getTime()
    if ModeNum != 0:
        ModalData = {'Step': []}
        for i in range(ModeNum):
            ModalData[f'T{i}'] = []
            ModalData[f'RatX{i}'] = []
            ModalData[f'RatY{i}'] = []
            ModalData[f'RatR{i}'] = []
    while (CurTime <= Duration) and (THAFlag == 0):
        # print(f'{Duration} | {CurTime}')
        THAFlag = ops.analyze(1, dt)
        CurTime = ops.getTime()
        if ModeNum != 0:
            _ = ops.eigen(ModeNum)
            ModeOuts = ops.modalProperties('-return')
            ModalData['Step'].append(CurTime)
            for i in range(ModeNum):
                ModalData[f'T{i}'].append(ModeOuts['eigenPeriod'][i])
                ModalData[f'RatX{i}'].append(ModeOuts['partiMassRatiosMX'][i])
                ModalData[f'RatY{i}'].append(ModeOuts['partiMassRatiosMY'][i])
                ModalData[f'RatR{i}'].append(ModeOuts['partiMassRatiosRMZ'][i])

        if THAFlag != 0:
            # print(f'Fail at {CurTime} 0.5*dt being applied...')
            dtCur = 0.5*dt
            THAFlag = ops.analyze(1, dtCur)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} 0.25*dt being applied...')
            dtCur = 0.25*dt
            THAFlag = ops.analyze(1, dtCur)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} 0.5*dt being applied...')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10, 1)
            dtCur = 0.5*dt
            THAFlag = ops.analyze(1, dtCur)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} 0.25*dt being applied...')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10, 1)
            dtCur = 0.25*dt
            THAFlag = ops.analyze(1, dtCur)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying Broyden...')
            ops.algorithm('Broyden', *['-count', True])
            THAFlag = ops.analyze(1, dt)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying Newton with Initial Tangent...')
            ops.algorithm('Newton', *['-initial', True])
            THAFlag = ops.analyze(1, dt)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying NewtonWithLineSearch...')
            ops.algorithm('NewtonLineSearch')
            THAFlag = ops.analyze(1, dt)
            ops.algorithm(algorithmTyp)

        # Now change algorithm with tol
        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying Newton with Initial Tangent'
            #       f'Tolerance= {tolInit*0.1} IterNum= {iterInit*10}...')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10)
            ops.algorithm('Newton', *['-initial', True])
            THAFlag = ops.analyze(1, dt)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying NewtonWithLineSearch Tolerance='
            #       f'{tolInit*0.1} IterNum= {iterInit*10}......')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10)
            ops.algorithm('NewtonLineSearch')
            THAFlag = ops.analyze(1, dt)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)

        # Maybe timestep + convergence + alg
        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying Newton with Initial Tangent '
            #       f'Tolerance= {tolInit*0.1} IterNum= {iterInit*10}'
            #       f'dt = {dt*0.5}...')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10)
            ops.algorithm('Newton', *['-initial', True])
            dtCur = 0.5*dt
            THAFlag = ops.analyze(1, dtCur)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)

        if THAFlag != 0:
            # print(f'Fail at {CurTime} Trying NewtonWithLineSearch Tolerance='
            #       f'{tolInit*0.1} IterNum= {iterInit*10} dt = {dt*0.5}......')
            ops.test('NormDispIncr', tolInit*0.1, iterInit*10)
            ops.algorithm('NewtonLineSearch')
            dtCur = 0.5*dt
            THAFlag = ops.analyze(1, dtCur)
            THAFlag = ops.analyze(1, dt)
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm(algorithmTyp)
        CurTime = ops.getTime()

    if CurTime < Duration:
        ReturnFlag = 99
    else:
        ReturnFlag = 0
    if ModeNum != 0:
        PeriodChange = np.array([np.array(ModalData[k])
                                 for k in ModalData.keys()])
        np.savetxt('PeriodChange.csv', PeriodChange)
    return ReturnFlag


def do(Accelerations, dt,
       Duration=0,
       DampingInfo={'Rayleigh': {'i': 1,
                                 'j': 3,
                                 'xii': 0.05,
                                 'xij': 0.05}}):
    if 'Rayleigh' in DampingInfo:
        L = ops.eigen(DampingInfo['j'])
        omegas = np.sqrt(L)
        wi = omegas[DampingInfo['i']-1]
        wj = omegas[DampingInfo['j']-1]
        A = np.array([[1/wi, wi], [1/wj, wj]])
        b = np.array([DampingInfo['xii'],
                      DampingInfo['xij']])
        a = np.linalg.solve(A, 2*b)
        ops.wipeAnalysis()
        ops.rayleigh(a[0], 0.0, 0.0, a[1])
    elif 'Modal' in DampingInfo:
        ops.wipeAnalysis()
        ops.eigen(DampingInfo['Modal']['i'])
        ops.modalDamping(DampingInfo['Modal']['xi'])

    # Define Timeseries
    AccX = Accelerations[0]
    AccY = Accelerations[1]
    THxTag = 700
    THyTag = 701

    ops.timeSeries('Path', THxTag,
                   '-values', *list(AccX), '-dt', dt, '-factor', 1.0)
    ops.timeSeries('Path', THyTag,
                   '-values', *list(AccY), '-dt', dt, '-factor', 1.0)

    # Define Pattern
    PxTag = 710
    PyTag = 711

    # pattern('UniformExcitation', patternTag, dir,
    # '-disp', dispSeriesTag, '-vel', velSeriesTag,
    # '-accel', accelSeriesTag, '-vel0', vel0, '-fact', fact)
    ops.pattern('UniformExcitation', PxTag, 1, '-accel', THxTag)
    ops.pattern('UniformExcitation', PyTag, 2, '-accel', THyTag)

    if len(Accelerations) == 3:
        THzTag = 702
        PzTag = 712
        AccZ = Accelerations[2]
        ops.timeSeries('Path', THzTag, '-values', *list(AccZ),
                       '-dt', dt, '-factor', 1.0)
        ops.pattern('UniformExcitation', PzTag, 2, '-accel', THzTag)

    # Define analysis parameters
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('FullGeneral')

    if Duration == 0:
        Duration = len(AccX)*dt
    # ReturnFlag = _doSingleRHA(Duration, dt)
    ReturnFlag = _doSingleRHA(Duration, dt)
    return ReturnFlag

