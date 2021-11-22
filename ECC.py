# 电卡效应制冷循环一维数值模型
# 文件名：ece_1d_sim.py
# 创建日期：2021.10.26

import numpy as np 
import sympy as sp

# 热沉材料属性，铜
k_sink = 395 	# 导热系数，单位：W/m·K
ρ_sink = 8930	# 密度，单位：kg/m³
c_sink = 383 	# 比热容，单位：J/kg·K
d_sink = 5e-3	# 厚度，单位：m

# 热源材料属性，铜
k_src = 395 	# 导热系数，单位：W/m·K
ρ_src = 8930	# 密度，单位：kg/m³
c_src = 383 	# 比热容，单位：J/kg·K
d_src = 5e-3	# 厚度，单位：m

# 电卡材料属性，P(VDF-TrFE-CFE)
k_pvdf = 0.2 	# 导热系数，单位：W/m·K
ρ_pvdf = 1800	# 密度，单位：kg/m³
c_pvdf = 1500	# 比热容，单位：J/kg·K
d_pvdf = 1e-4	# 厚度，单位：m

# 边界条件
Tsurr = 290		# 环境温度，单位：K
heff = 300		# 散热器强制对流换热系数，单位：W/(m²·K)
R = 2e-4		# 电卡与热沉（热源）间的接触热阻，单位：m²·K/W
qgen = 1000		# 单位面积芯片产热功率，单位：W/m²

# 空间离散
n = 10 			# 空间离散，每层离散为10份
m = 3*n			# 三层结构的微元总数
e = 1e-2			# 稳态判据，相邻迭代的温度差异低于0.01K认为达到稳态
simStep = 1 		# 仿真时间序号

# 创建A*T=B方程的三个矩阵
A = np.zeros([m, m], dtype=float) 	# 参数矩阵A
B = np.zeros([m, 1], dtype=float) 	# 上一时刻的温度矩阵（含热源项）
T = np.ones([m, 1], dtype=float)	# 下一时刻的温度矩阵
T_prev 	= np.zeros([m,1], dtype=float) # 中间过程量，用于暂存温度数据
T_error	= np.ones([m,1], dtype=float) # 温度差异

# 实际时间和仿真时间
t1 = 0.1 				# 电场加载（卸载）时长
t2 = 4 					# 吸热（放热）时长
Δt = 0.001  			# 时间步长1ms
cycTime = (t1+t2)*2 	# 单个电卡循环的时间
cycStep = cycTime/Δt 	# 单个循环内的步长总数 

q = qgen*Δt/(ρ_src*c_src)

def calculateA():
	global A, simStep

	# 实际时间和仿真时间
	t1Step = t1/Δt 			# 单个循环内电场加载（卸载）步长总数
	t2Step = t2/Δt 			# 单个循环内吸热（放热）步长总数
	Δx1 = d_sink/n 		# 热源、热沉微元厚度
	Δx2 = d_pvdf/n 		# 电卡材料微元厚度

	# 确定4个阶段stage的参数矩阵A
	cycIndex = simStep%cycStep
	# stage取值如下：
	if cycIndex <= t1Step:
		stage = 1 		# stage = 1，加载电场，升温
	if cycIndex <= (t1Step+t2Step) and cycIndex > t1Step:
		stage = 2 		# stage = 2，接触热沉，放热
	if cycIndex <= (2*t1Step+t2Step) and cycIndex > (t1Step+t2Step):
		stage = 3 		# stage = 3，去除电场，降温
	if cycIndex <= cycStep and cycIndex > (2*t1Step+t2Step):
		stage = 4		# stage = 4，接触热源，吸热
	# 热沉散热边界层系数a11,a12
	a11 = 1 + k_sink/(heff*Δx1)
	a12 = -k_sink/(heff*Δx1)
	
	# 确定以b开头的系数，对三层中的每一层内部传热过程分开讨论
	# 热沉层内部导热过程系数
	b11 = -k_sink*Δt/(ρ_sink*c_sink*Δx1**2)
	b12 = 1+2*k_sink*Δt/(ρ_sink*c_sink*Δx1**2)
	b13 = b11

	# 热源层内部导热过程系数
	b31 = -k_src*Δt/(ρ_src*c_src*Δx1**2)
	b32 = 1+2*k_src*Δt/(ρ_src*c_src*Δx1**2)
	b33 = b31

	# 电卡材料内部传热过程系数，分阶段讨论
	b21 = -k_pvdf*Δt/(ρ_pvdf*c_pvdf*Δx2**2)
	b23 = b21
	# 计算K，涉及到电卡材料熵变随时间变化关系
	t = (simStep%cycStep)*Δt
	if stage == 1:
		E = 1e8/t1*t 	# t1为电卡加载电场的时长0.1s, 1e8为电场强度100Mv/m
		P = -1.20626e-4+4.42499e-10*E-1.87543e-18* E**2+2.2441e-26* E**3+3.96694e-37*E**4
		c1 = -2.71e-15
		c2 = -6.85e-8
		E = sp.symbols('E')
		PE = sp.diff(P,E)
		K = ρ_pvdf*2*(c1*E**2+c2*E)/P*sp.diff(PE,E).subs(E,1e8/t1*t)*1e8/t1
	if stage == 2:
		K = 0
	if stage == 3:
		E = 1e8/t1*(2*t1+t2-t)
		P = 0.00531+6.5182e-10*E+4.71971e-18*E**2-1.13175e-26*E**3-2.29666e-34* E**4
		c1 = -2.71e-15
		c2 = -6.85e-8
		E = sp.symbols('E')
		PE = sp.diff(P,E)
		K = ρ_pvdf*2*(c1*E**2+c2*E)/P*sp.diff(PE,E).subs(E,1e8/t1*(2*t1+t2-t))*1e8/t1
	if stage == 4:
		K = 0
	# b22 = 1 + 2*k_pvdf*Δt/(ρ_pvdf*c_pvdf*Δx2**2 + K) # 此处有误
	b22 = 1 + 2*k_pvdf*Δt/(ρ_pvdf*c_pvdf*Δx2**2) - k_pvdf*Δt/(ρ_pvdf*c_pvdf)


	# 确定以c开头的系数，接触传热分阶段讨论
	# 电场加载、卸载阶段
	if stage == 1 or stage == 3:
		# 电卡与热沉不接触
		c11 = -1
		c12 = 1
		c13 = 0 	# 额外添加
		c21 = 0 	# 额外添加
		c22 = -1
		c23 = 1
		# 电卡与热源不接触
		c31 = -1
		c32 = 1
		c33 = 0 	# 额外添加
		c41 = 0 	# 额外添加
		c42 = -1
		c43 = 1
	# 放热阶段
	if stage == 2:
		# 电卡与热沉接触
		c11 = -k_sink*R/Δx1
		c12 = 1 + k_sink*R/Δx1
		c13 = -1 	# 额外添加
		c21 = 1 	# 额外添加
		c22 = -1 - k_pvdf*R/Δx2
		c23 = k_pvdf*R/Δx2
		# 电卡与热源不接触
		c31 = -1
		c32 = 1
		c33 = 0 	# 额外添加
		c41 = 0 	# 额外添加
		c42 = -1
		c43 = 1
	# 吸热阶段
	if stage == 4:
		# 电卡与热沉不接触
		c11 = -1
		c12 = 1
		c13 = 0 	# 额外添加
		c21 = 0 	# 额外添加
		c22 = -1
		c23 = 1
		# 电卡与热源接触
		c31 = -k_pvdf*R/Δx2
		c32 =  1 + k_pvdf*R/Δx2
		c33 = -1 	# 额外添加
		c41 = 1 	# 额外添加
		c42 = -1 - k_src*R/Δx1
		c43 = k_src*R/Δx1

	# 对参数矩阵A中的3主对角线值赋值
	A[0,0] = a11
	A[0,1] = a12
	for index in range(1,n-1):
		A[index,index] = b12
		A[index,index-1] = b11
		A[index,index+1] = b11
	A[n-1,n-2]	= c11
	A[n-1,n-1] 	= c12
	A[n-1,n]	= c13
	A[n,n-1] 	= c21
	A[n,n] 		= c22
	A[n,n+1] 	= c23
	for index in range(n+1,2*n-1):
		A[index,index] = b22
		A[index,index-1] = b21
		A[index,index+1] = b21
	A[2*n-1,2*n-2]	= c31
	A[2*n-1,2*n-1] 	= c32
	A[2*n-1,2*n]	= c33
	A[2*n,2*n-1] 	= c41
	A[2*n,2*n] 		= c42
	A[2*n,2*n+1] 	= c43
	for index in range(2*n+1,3*n-1):
		A[index,index] = b32
		A[index,index-1] = b31
		A[index,index+1] = b31
	A[m-1,m-2] = 1
	A[m-1,m-1] = -1
	
def initB():
	# 温度矩阵B初始化，即三层材料所有节点温度初始化
	global B
	B[0,0] = Tsurr
	B[n-1,0] = 0
	B[n,0] = 0
	B[2*n-1,0] = 0
	B[2*n,0] = 0
	B[m-1,0] = 0
	for i in range(1,n-1):
		B[i,0] = Tsurr 	# 假设热沉内部初始温度均为环境温度
	for i in range(n+1,2*n-1):
		B[i,0] = Tsurr 	# 假设电卡内部初始温度均为环境温度
	for i in range(2*n+1,m-1):
		B[i,0] = 310 	# 假设热源初始温度比环境温度高20K，即310K

def converge():
	# 各点温度趋于稳定的收敛判据
	global T, T_prev, T_error, simStep
	if (simStep%cycStep == 0):
		T_error = T - T_prev
		T_prev = T
		if (np.array(T_error)<e).all() and (np.array(T_error)>-e).all():
			return True
		else:
			return False

def iterationT():
	global A, B, T, simStep, q
	# 对矩阵A取逆，同时在A*T=B方程两侧左乘A逆，得到下一时刻的温度矩阵T
	T = np.dot((np.linalg.inv(A)),B)
	# 将本次计算的T赋值给B
	for i in range(0,2*n):
		B[i,0] = T[i,0]
	for i in range(2*n+1,m):
		B[i,0] = T[i,0]+q # 此处有误，应将qgen改为q
	B[0,0] = Tsurr
	B[n-1,0] = 0
	B[n,0] = 0
	B[2*n-1,0] = 0
	B[2*n,0] = 0
	B[m-1,0] = 0
	# 迭代下一步
	simStep += 1

if __name__ == '__main__':
	initB()
	while not converge():
		calculateA()
		iterationT()
		print(simStep)
		print(T)
		print(T_error)