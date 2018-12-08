#### 03.1.数组中重复的数字
    在一个长度为n的数组里的所有数字都在0到n-1的范围内。 
    数组中某些数字是重复的，但不知道有几个数字是重复的。
    也不知道每个数字重复几次。请找出数组中任意一个重复的数字。
    例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
##### 思路
    一
        1.将数组进行排序
        2.扫描数组即可确定重复数字
        3.排序数组时间复杂度O(nlogn),扫描数组O(n)
    二
        1.扫描数组，将每个数字与hash表进行对比
        2.如果hash表没有则把它添加进去，如果存在就找到一个重复数组
        3.时间复杂度O(1)，空间复杂度O(n)
    三
        1.

##### Code
```
//Golang
func duplicate(numbers []int, length int, duplication *int) bool {
	if numbers == nil || length <= 0 {
		return false
	}
	for i := 0; i < length; i++ {
		if numbers[i] < 0 || numbers[i] > length-1 {
			return false
		}
	}
	for i := 0; i < length; i++ {
		for numbers[i] != i {
			if numbers[i] == numbers[numbers[i]] {
				*duplication = numbers[i]
				return false
			}
			temp := numbers[i]
			numbers[i] = numbers[temp]
			numbers[temp] = temp
		}
	}
	return false
}
```
#### 03.2.不修改数组中重复的数字
    在一个长度为n+1的数组里的所有数字都在1~n的范围内。 
    所以数组中至少有一个数字是重复的。
    请找出数组中任意一个重复的数字，但不能修改输入的数组。
    例如，如果输入长度为8的数组{2,3,5,4,3,2,6,7}，那么对应的输出是第一个重复的数字2或者3。
##### 思路


##### Code
```
//Golang
func getDuplications(numbers *[]int, length int) int {
	if numbers == nil || length <= 0 {
		return -1
	}
	start := 1
	end := length - 1
	for end >= start {
		middle := ((end - start) >> 1) + start
		count := countRange(numbers, length, start, middle)
		if end == start {
			if count > 1 {
				return start
			} else {
				break
			}
		}
		if count > (middle - start + 1) {
			end = middle
		} else {
			start = middle + 1
		}
	}
	return -1
}

func countRange(numbers *[]int, length int, start int, end int) int {
	if &numbers == nil {
		return 0
	}
	count := 0
	for i := 0; i < length; i++ {
		if (*numbers)[i] >= start && (*numbers)[i] <= end {
			count++
		}
	}
	return count
}
```

#### 04.二维数组中的查找
    在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
    每一列都按照从上到下递增的顺序排序。请完成一个函数，
    输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
##### 思路
1.     选取数组右上角数， 如果该数等于要查找的数，查找结束
2.     如果该数大于要查找的数，则剔除该数所在的列
3.     如果该数小于要查找的数，则剔除该数所在的行
4.     直到找到要查找的数字，或者查找为空
##### Code
```
func Find(matrix [][]int, rows, columns, number int) bool {
	found := false
	if matrix != nil && rows > 0 && columns > 0 {
		row := 0
		column := columns - 1
		for row < rows && column >= 0 {
			if matrix[row][column] == number {
				found = true
				break
			} else if matrix[row][column] > number {
				column--
			} else {
				row++
			}
		}
	}
	return found
}
```
#### 05.替换空格
    题目：请实现一个函数，把字符串中的每个空格替换成"%20"。例如输入“We are happy.”，则输出“We%20are%20happy.”。
##### 思路一
1.  把字符串第一个空格替换， 后面字符整体后移
2.  循环执行第一步，直到字符串没有空格
3.  时间复杂度O(n^2)
##### 思路二
1.  先获取字符串空格数，并计算替换后字符串总长度
2.  新建新数组，遍历之前的数组，如有空格进行字符替换
3.  时间复杂度O(n)
##### Code
```
func ReplaceBlank(str string) string {
	if len(str) == 0 {
		return str
	}
	numOfSpaces := countSpace(str)
	finalLength := len(str) + 2*numOfSpaces
	resultString := make([]rune, finalLength)
	j := 0
	for _, r := range str {
		if r == ' ' {
			resultString[j] = '%'
			j++
			resultString[j] = '2'
			j++
			resultString[j] = '0'
			j++
		} else {
			resultString[j] = r
			j++
		}
	}
	return string(resultString)
}

func countSpace(str string) int {
	num := 0
	for _, r := range str {
		if r == ' ' {
			num++
		}
	}
	return num
}
```
#### 06.从尾到头打印链表
    题目：输入一个链表的头结点，从尾到头反过来打印出每个结点的值。
##### 思路
##### Code
```
type ListNode struct {
	value int
	next  *ListNode
}

// 添加元素节点
func (l *ListNode) append(e int) *ListNode {
	node := ListNode{e, nil}
	ptr := l
	for ptr.next != nil {
		ptr = ptr.next
	}
	ptr.next = &node
	return l
}

// 非递归方式
func (l *ListNode) PrintListReverse() {
	// 先存储在，然后再反序列打印
	var slice []int
	p := l
	for p != nil {
		//println(l.value)
		slice = append(slice, p.value)
		p = p.next
	}
	//println(slice)
	for i := len(slice) - 1; i >= 0; i-- {
		fmt.Print(slice[i])
	}
}

// 递归方式
func PrintListReverse1(ptr *ListNode) {
	if ptr != nil {
		if ptr.next != nil {
			PrintListReverse1(ptr.next)
		}
		fmt.Print(ptr.value)
	}
}
```
#### 07.重建二叉树
    题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
    假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2, 4, 7, 3, 5, 6, 8}和中序遍历序列{4, 7, 2, 1, 5, 3, 8, 6}，则重建出图2.6所示的二叉树并输出它的头结点。
##### 思路
##### Code
```
// 二叉树结构
type BTree struct {
	lchild *BTree
	value  int
	rchild *BTree
}

// 后序遍历
func (t *BTree) Inorder() {
	if t != nil {
		t.lchild.Inorder()
		fmt.Print(t.value)
		t.rchild.Inorder()
	}
}

// 通过前序后序、序列构建二叉树
// inOrder中序序列 preOrder前序序列
func Construct(preOrder []int, inOrder []int) *BTree {
	// ConstructCore函数的封装，只需用户提供序列就可以构建二叉树
	if preOrder == nil || inOrder == nil {
		return nil
	}
	return ConstructCore(preOrder, 0, len(preOrder)-1,
		inOrder, 0, len(inOrder)-1)
}

// 传入preOrder[]先序，中序inOrder[]序列，用startPreOrder和endPreOrder标记起始位置和终止位置
func ConstructCore(preOrder []int, startPreOrder int, endPreOrder int,
	inOrder []int, startInOrder int, endInOrder int) *BTree {

	// 前序遍历序列的第一个数字是根结点的值
	rootValue := preOrder[startPreOrder]
	root := &BTree{nil, rootValue, nil}

	if startPreOrder == endPreOrder {
		if startInOrder == endInOrder &&
			preOrder[startPreOrder] == inOrder[startInOrder] {
			return root
		} else {
			fmt.Println("Invalid input!")
		}
	}

	// 在中序遍历中找到根结点的值
	rootInOrder := startInOrder
	for rootInOrder <= endInOrder && inOrder[rootInOrder] != rootValue {
		rootInOrder++
	}

	// 输入的两个序列不匹配的情况
	if rootInOrder == endInOrder && inOrder[rootInOrder] != rootValue {
		fmt.Println("Invalid input!")
	}

	leftLength := rootInOrder - startInOrder
	leftPreOrderEnd := startPreOrder + leftLength
	if leftLength > 0 {
		// 构建左子树
		root.lchild = ConstructCore(preOrder, startPreOrder+1, leftPreOrderEnd,
			inOrder, startInOrder, rootInOrder-1)
	}
	if leftLength < endPreOrder-startPreOrder {
		// 构建右子树
		root.rchild = ConstructCore(preOrder, leftPreOrderEnd+1, endPreOrder,
			inOrder, rootInOrder+1, endInOrder)
	}
	return root
}
```
#### 08.二叉树的下一个节点
    题目：给定一棵二叉树和其中的一个结点，如何找出中序遍历顺序的下一个结点？
    树中的结点除了有两个分别指向左右子结点的指针以外，还有一个指向父结点的指针。
##### 思路
##### Code
```
```
#### 09.用两个栈实现队列
    题目：用两个栈实现一个队列。队列的声明如下，请实现它的两个函数appendTail和deleteHead，分别完成在队列尾部插入结点和在队列头部删除结点的功能。
##### 思路
##### Code
```
import (
	"errors"
	"fmt"
	"stack"
)

type queue struct {
	s1 stack.Stack
	s2 stack.Stack
}

func (q *queue) Pop() interface{} {
	// 判断s2是否为空，如不为空，则直接弹出顶元素；如为空，则将s1的元素逐个“倒入”s2，把s2栈顶元素弹出并出队。
	if q.s2.IsEmpty() {
		size := q.s1.Size()
		for i := 0; i < size; i++ {
			x, _ := q.s1.Pop()
			q.s2.Push(x)
		}
	}
	res, _ := q.s2.Pop()

	// 空接口类型，可用断言的方式获取当前类型输出
	switch res.(type) {
	case int:
		return res.(int)
	case string:
		return res.(string)
	case nil:
		res = errors.New("queue is empty")
	default:
		res = errors.New("Unexpected type")
	}
	return res
}

func (q *queue) Push(e interface{}) {
	// 入队时，将元素压入s1。
	q.s1.Push(e)
}
```
#### 10.斐波那契数列
    题目：写一个函数，输入n，求斐波那契（Fibonacci）数列的第n项。
##### 思路

##### Code
```
// Golang
// 递归解法
func Fibonacci1(n uint) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	return Fibonacci1(n-1) + Fibonacci1(n-2)
}

// 循环解法
func Fibonacci2(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	One, Two := 1, 0
	fibN := 0
	for i := 2; i <= n; i++ {
		fibN = One + Two
		Two = One
		One = fibN
	}
	return fibN
}
```

#### 11.旋转数组的最小数字
    题目：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
    例如数组{3, 4, 5, 1, 2}为{1, 2, 3, 4,5}的一个旋转，该数组的最小值为1。
##### 思路
##### Code
```
// Golang
func Min(numbers []*int, length int) int {
	if numbers == nil || length <= 0 {
		return -1
	}
	index1 := 0
	index2 := length - 1
	indexMid := index1
	for *numbers[index1] >= *numbers[index2] {
		if index2-index1 == 1 {
			indexMid = index2
			break
		}
		indexMid = (index1 + index2) / 2
		if numbers[index1] == numbers[index2] && numbers[indexMid] == numbers[index1] {
			return MinInOrder(numbers, index1, index2)
		}
		if *numbers[indexMid] >= *numbers[index1] {
			index1 = indexMid
		} else if *numbers[indexMid] <= *numbers[index2] {
			index2 = indexMid
		}
	}
	return indexMid
}

func MinInOrder(numbers []*int, index1, index2 int) int {
	result := *numbers[index1]
	for i := index1 + 1; i <= index2; i++ {
		if result > *numbers[i] {
			result = *numbers[i]
		}
	}
	return result
}
```
#### 12.矩阵中的路径
    题目：请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用下划线标出）。但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
    A B T G
    C F C S
    J D E H
##### 思路
##### Code
```
```
#### 13.机器人的运动范围
    题目：地上有一个m行n列的方格。一个机器人从坐标(0, 0)的格子开始移动，它每一次可以向左、右、上、下移动一格，但不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格(35, 37)，因为3+5+3+7=18。但它不能进入方格(35, 38)，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
##### 思路
##### Code
```
```
#### 14.剪绳子
    题目：给你一根长度为n绳子，请把绳子剪成m段（m、n都是整数，n>1并且m≥1）。
    每段的绳子的长度记为k[0]、k[1]、……、k[m]。k[0]*k[1]*…*k[m]可能的最大乘积是多少？
    例如当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到最大的乘积18。
##### 思路
##### Code
```
```
#### 15.二进制的1的个数
    题目：请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。
    例如把9表示成二进制是1001，有2位是1。因此如果输入9，该函数输出2。
##### 思路
##### Code
```
func NumberOfOne1(n int) int {
	count := 0
	for n != 0 { // bool(n)编译错误，go语言中1不等同于ture。
		if (n & 1) != 0 {
			count++
		}
		n = n >> 1
	}
	return count
}

func NumberOfOne2(n int) int {
	flag := 1
	count := 0
	for flag != 0 {
		if (n & flag) != 0 {
			count++
		}
		flag = flag << 1
	}
	return count
}

func NumberOfOne3(n int) int {
	count := 0
	for n != 0 {
		count++
		n = (n - 1) & n

	}
	return count
}
```
#### 16.数值的整数次方
    题目：实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
##### 思路
##### Code
```
```
#### 17.打印1到最大的n位数
    题目：输入数字n，按顺序打印出从1最大的n位十进制数。比如输入3，
    则打印出1、2、3一直到最大的3位数即999。
##### 思路
##### Code
```
```
#### 18.1在O(1)时间删除链表节点
    题目：给定单向链表的头指针和一个结点指针，定义一个函数在O(1)时间删除该结点。
##### 思路
##### Code
```
```
#### 18.2删除链表中重复的节点
    题目：在一个排序的链表中，如何删除重复的结点？例如，在图3.4（a）中重复结点被删除之后，链表如图3.4（b）所示。
##### 思路
##### Code
```
```
#### 19.正则表达式匹配
    题目：请实现一个函数用来匹配包含'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"及"ab*a"均不匹配。
##### 思路
##### Code
```
```
#### 20.表示数值的字符串
    题目：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
    例如，字符串“+100”、“5e2”、“-123”、“3.1416”及“-1E-16”都表示数值，
    但“12e”、“1a3.14”、“1.2.3”、“+-5”及“12e+5.4”都不是
##### 思路
##### Code
```
```
#### 21.调整数组顺序使奇数位于偶数前面
    题目：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
    使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
##### 思路
##### Code
```
```
#### 22.链表中倒数第K个节点
    题目：输入一个链表，输出该链表中倒数第k个结点。为了符合大多数人的习惯，
    本题从1开始计数，即链表的尾结点是倒数第1个结点。例如一个链表有6个结点，
    从头结点开始它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个结点是值为4的结点。
##### 思路
##### Code
```
```
#### 23.链表中环的入口节点
    题目：一个链表中包含环，如何找出环的入口结点？例如，在图3.8的链表中，
    环的入口结点是结点3。
##### 思路
##### Code
```
```
#### 24.反转链表
    题目：定义一个函数，输入一个链表的头结点，反转该链表并输出反转后链表的头结点。
##### 思路
##### Code
```
```
#### 25.合并两个排序的链表
    题目：输入两个递增排序的链表，合并这两个链表并使新链表中的结点仍然是
    按照递增排序的。例如输入图3.11中的链表1和链表2，则合并之后的升序链表如链表3所示。
##### 思路
##### Code
```
```
#### 26.树的子结构
    题目：输入两棵二叉树A和B，判断B是不是A的子结构。
##### 思路
##### Code
```
```
#### 27.二叉树的镜像
    题目：请完成一个函数，输入一个二叉树，该函数输出它的镜像。
##### 思路
##### Code
```
```
#### 28.对称的二叉树
    题目：请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，
    那么它是对称的。
##### 思路
##### Code
```
```
#### 29.顺时针打印矩阵
    题目：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
##### 思路
##### Code
```
```
#### 30.包含min函数的栈
    题目：定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的min函数。
    在该栈中，调用min、push及pop的时间复杂度都是O(1)。
##### 思路
##### Code
```
import (
	"github.com/emirpasic/gods/stacks/arraystack"
	"errors"
)

var (
	dataStack = arraystack.New()
	minStack  = arraystack.New()
)

func push(number int) {
	dataStack.Push(number)

	if 0 == minStack.Size() {
		minStack.Push(number)
	} else {
		topNumber, _ := minStack.Peek()
		topN := topNumber.(int)

		if number < topN {
			minStack.Push(number)
		} else {
			minStack.Push(topN)
		}
	}
}

func pop() {
	if 0 < dataStack.Size() && 0 < minStack.Size() {
		dataStack.Pop()
		minStack.Pop()
	}
}

func min() (int, error) {
	if 0 < dataStack.Size() && 0 < minStack.Size() {
		topNumber, _ := minStack.Pop()
		topN := topNumber.(int)
		return topN, nil
	}
	return 0, errors.New("栈为空")
}
```
#### 31.栈的压入，弹出序列
    题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
    假设压入栈的所有数字均不相等。例如序列1、2、3、4、5是某栈的压栈序列，
    序列4、5、3、2、1是该压栈序列对应的一个弹出序列，
    但4、3、5、1、2就不可能是该压栈序列的弹出序列。
##### 思路
##### Code
```
import "github.com/emirpasic/gods/stacks/arraystack"

func isPopOrder(pushStack, popStack []int) bool {
	length := len(pushStack)
	if 0 == length || length != len(popStack) {
		return false
	}

	stack := arraystack.New()
	nextPush, nextPop := 0, 0

	for nextPop < length {
		for topNumber, _ := stack.Peek(); nil == topNumber || topNumber != popStack[nextPop]; {
			if nextPush == length {
				break
			}
			stack.Push(pushStack[nextPush])
			nextPush++

			topNumber, _ = stack.Peek()
		}
		if topNumber, _ := stack.Peek(); topNumber != popStack[nextPop] {
			break
		}

		stack.Pop()
		nextPop++
	}

	if stack.Empty() && nextPop == length {
		return true
	}
	return false
}
```
#### 32.1.不分行从上往下打印二叉树
    题目：从上往下打印出二叉树的每个结点，同一层的结点按照从左到右的顺序打印。
##### 思路
##### Code
```
import (
	"gopkg.in/oleiade/lane.v1"
	"fmt"
)

type binaryTreeNode struct {
	value int
	left  *binaryTreeNode
	right *binaryTreeNode
}

func breadthFirstSearch(root *binaryTreeNode) {
	if nil == root {
		return
	}
	var deque = lane.NewDeque()
	deque.Append(root)

	for !deque.Empty() {
		ptr := deque.Shift().(*binaryTreeNode)
		fmt.Println(ptr.value)
		if ptr.left != nil {
			deque.Append(ptr.left)
		}
		if ptr.right != nil {
			deque.Append(ptr.right)
		}
	}
}

// 分行打印二叉树
func breadthFirstSearch2(root *binaryTreeNode) {
	if nil == root {
		return
	}
	var deque = lane.NewDeque()
	deque.Append(root)
	// 当前层中还没有打印的节点数
	currentLevel := 1
	// 下一层的节点数
	nextLevel := 0
	for !deque.Empty() {
		ptr := deque.First().(*binaryTreeNode)
		fmt.Print(ptr.value, " ")
		if ptr.left != nil {
			deque.Append(ptr.left)
			nextLevel++
		}
		if ptr.right != nil {
			deque.Append(ptr.right)
			nextLevel++
		}
		deque.Shift()
		currentLevel--
		if 0 == currentLevel {
			currentLevel = nextLevel
			nextLevel = 0
			fmt.Printf("\n")
		}
	}
}
```
#### 32.1.不分行从上往下打印二叉树
    题目：从上往下打印出二叉树的每个结点，同一层的结点按照从左到右的顺序打印。
##### 思路
##### Code
```
// begin 开始字符的序号， end 结尾字符的序号
func verifySquenceOfBST(sequence []int, begin, end int) bool {
	if nil == sequence || begin >= end {
		return false
	}

	root := sequence[end]

	// 二叉搜索树中，左子树节点的值小于根节点的值
	i := begin
	for ; i < end; i++ {
		if sequence[i] > root {
			break
		}
	}
	// 二叉搜索树中，右子树节点的值大于根节点的值
	j := i
	for ; j < end; j++ {
		if sequence[j] < root {
			return false
		}
	}
	// 判断左子树是不是二叉搜索树
	left := true
	if 1 < i-begin {
		left = verifySquenceOfBST(sequence, begin, i-1)
	}
	// 判断右子树是不是二叉搜索树
	right := true
	if 1 < end-i {
		right = verifySquenceOfBST(sequence, i, end-1)
	}

	return left && right
}
```
#### 32.2.不分行从上往下打印二叉树
    题目：从上到下按层打印二叉树，同一层的结点按从左到右的顺序打印，
    每一层打印到一行。
##### 思路
##### Code
```
```
#### 32.3.之字形打印二叉树
    题目：请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，
    第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
##### 思路
##### Code
```
```
#### 33.二叉搜索树的后序遍历序列
    题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
    如果是则返回true，否则返回false。假设输入的数组的任意两个数字都互不相同。
##### 思路
##### Code
```
// begin 开始字符的序号， end 结尾字符的序号
func verifySquenceOfBST(sequence []int, begin, end int) bool {
	if nil == sequence || begin >= end {
		return false
	}

	root := sequence[end]

	// 二叉搜索树中，左子树节点的值小于根节点的值
	i := begin
	for ; i < end; i++ {
		if sequence[i] > root {
			break
		}
	}
	// 二叉搜索树中，右子树节点的值大于根节点的值
	j := i
	for ; j < end; j++ {
		if sequence[j] < root {
			return false
		}
	}
	// 判断左子树是不是二叉搜索树
	left := true
	if 1 < i-begin {
		left = verifySquenceOfBST(sequence, begin, i-1)
	}
	// 判断右子树是不是二叉搜索树
	right := true
	if 1 < end-i {
		right = verifySquenceOfBST(sequence, i, end-1)
	}

	return left && right
}
```
#### 34.二叉树中和为某一值的路径
    题目：输入一棵二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
    从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
##### 思路
##### Code
```
import (
	"github.com/emirpasic/gods/stacks/arraystack"
	"fmt"
)

type binaryTreeNode struct {
	value int
	left  *binaryTreeNode
	right *binaryTreeNode
}

func findPath(root *binaryTreeNode, expectedSum int) {
	if nil == root {
		return
	}
	stack := arraystack.New()
	currentSum := 0
	findPathCore(root, stack, expectedSum, currentSum)
}

func findPathCore(root *binaryTreeNode, stack *arraystack.Stack, expectedSum, currentSum int) {
	currentSum += root.value
	stack.Push(root.value)

	// 如果是叶节点，并且路径节点值的和等于输入的期望值则打印路径
	isLeaf := root.left == nil && root.right == nil
	if isLeaf && currentSum == expectedSum {
		it := stack.Iterator()
		for it.End(); it.Prev(); {
			fmt.Print(it.Value(), " ")
		}
		fmt.Printf("\n")
	}

	// 如果不是叶节点，则遍历它的子节点
	if root.left != nil {
		findPathCore(root.left, stack, expectedSum, currentSum)
	}
	if root.right != nil {
		findPathCore(root.right, stack, expectedSum, currentSum)
	}
	stack.Pop()
}
```
#### 35.复杂链表的复制
    题目：请实现函数ComplexListNode* Clone(ComplexListNode* pHead)，
    复制一个复杂链表。在复杂链表中，每个结点除了有一个m_pNext指针指向下一个结点外，
    还有一个m_pSibling 指向链表中的任意结点或者nullptr。
##### 思路
##### Code
```
type complexListNode struct {
	value   int
	next    *complexListNode
	sibling *complexListNode
}

func cloneNodes(head *complexListNode) {
	pNode := head
	for nil != pNode {
		pCloned := new(complexListNode)
		pCloned.value = pNode.value
		pCloned.next = pNode.next
		pNode.next = pCloned
		pNode = pCloned.next
	}

}

func connectSiblingNodes(head *complexListNode) {
	pNode := head
	for nil != pNode {
		pCloned := pNode.next
		if nil != pNode.sibling {
			pCloned.sibling = pNode.sibling.next
		}
		pNode = pCloned.next
	}
}

func reconnectNodes(head *complexListNode) *complexListNode {
	pNode := head
	var pClonedHead, pClonedNode *complexListNode
	if nil != pNode {
		pClonedHead = pNode.next
		pClonedNode = pClonedHead
		pNode.next = pClonedNode.next
		pNode = pNode.next
	}

	for nil != pNode {
		pClonedNode.next = pNode.next
		pClonedNode = pClonedNode.next
		pNode.next = pClonedNode.next
		pNode = pNode.next
	}
	return pClonedHead
}

func clone(pHead *complexListNode) *complexListNode {
	cloneNodes(pHead)
	connectSiblingNodes(pHead)
	return reconnectNodes(pHead)
}
```
#### 36.二叉搜索树与双向链表
    题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
    要求不能创建任何新的结点，只能调整树中结点指针的指向。
##### 思路
##### Code
```
type binaryTreeNode struct {
	value       int
	left, right *binaryTreeNode
}

func conver(root *binaryTreeNode) *binaryTreeNode {
	var lastNodeInList *binaryTreeNode
	convertNode(root, &lastNodeInList)

	// lastNodeInList指向双向链表的尾节点，但是需要返回链表的头结点，所以进行下面操作。
	headOfList := lastNodeInList
	for nil != headOfList && nil != headOfList.left {
		headOfList = headOfList.left
	}
	return headOfList
}

func convertNode(node *binaryTreeNode, lastNodeInList **binaryTreeNode) {
	if nil == node {
		return
	}
	current := node
	if nil != current.left {
		convertNode(current.left, lastNodeInList)
	}
	current.left = *lastNodeInList
	if nil != *lastNodeInList {
		(*lastNodeInList).right = current
	}
	*lastNodeInList = current
	if nil != current.right {
		convertNode(current.right, lastNodeInList)
	}
}
```
#### 37.序列化二叉树
    题目：请实现两个函数，分别用来序列化和反序列化二叉树。
##### 思路
##### Code
```
```
#### 38.字符串的排列
    题目：输入一个字符串，打印出该字符串中字符的所有排列。例如输入字符串abc，
    则打印出由字符a、b、c所能排列出来的所有字符串abc、acb、bac、bca、cab和cba。
##### 思路
##### Code
```

func permutation(arrayChar []byte, start int) {
	if 1 >= len(arrayChar) {
		return
	}
	if start == len(arrayChar)-1 {
		for i := 0; i < len(arrayChar); i++ {
			fmt.Printf("%c", arrayChar[i])
		}
		fmt.Println()
	} else {
		for i := start; i < len(arrayChar); i++ {
			swap(arrayChar, start, i)
			permutation(arrayChar, start+1)
			swap(arrayChar, start, i)
		}
	}
}

func swap(array []byte, m, n int) {
	temp := array[m]
	array[m] = array[n]
	array[n] = temp
}
```
#### 39.数组中出现次数超过一半的数字
    题目：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    例如输入一个长度为9的数组{1, 2, 3, 2, 2, 2, 5, 4, 2}。 由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。
##### 思路
##### Code
```
func moreThanHalfNum(numbers []int) int {
	if nil == numbers || 0 >= len(numbers) {
		return 0
	}
	ans := numbers[0]
	times := 1
	for i := 1; i < len(numbers); i++ {
		if 0 == times {
			ans = numbers[i]
			times = 1
		} else if ans == numbers[i] {
			times++
		} else {
			times--
		}
	}
	return ans
}
```
#### 40.最小的k个数
    题目：输入n个整数，找出其中最小的k个数。例如输入4、5、1、6、2、7、3、8这8个数字，
    则最小的4个数字是1、2、3、4。
##### 思路
##### Code
```
func getLeastNumbers(data []int, k int) []int {
	if nil == data || k > len(data) {
		return nil
	}
	leastNumbers := make([]int, k)
	for i := 0; i < k; i++ {
		leastNumbers[i] = data[i]
	}
	sort.Ints(leastNumbers)

	for i:=k; i<len(data); i++ {
		if data[i] < leastNumbers[k-1] {
			leastNumbers[k-1] = data[i]
			sort.Ints(leastNumbers)
		}
	}
	return leastNumbers
}
```
#### 41.数据流中的中位数
    题目：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，
    那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，
    那么中位数就是所有数值排序之后中间两个数的平均值。
##### 思路
##### Code
```
```
#### 42.连续子数组的最大和
    题目：输入一个整型数组，数组里有正数也有负数。
    数组中一个或连续的多个整数组成一个子数组。求所有子数组的和的最大值。
    要求时间复杂度为O(n)。
##### 思路
##### Code
```
func findGreatestSumOfSubArray(data []int) (int, bool) {
	if nil == data || len(data) <= 0 {
		return 0, false
	}
	var currentSum, greatestSum int
	for i := 0; i < len(data); i++ {
		if currentSum <= 0 {
			currentSum = data[i]
		} else {
			currentSum += data[i]
		}
		if currentSum > greatestSum {
			greatestSum = currentSum
		}
	}
	return greatestSum, true
}
```

#### 43.从1到n整数中1出现的次数
    题目：输入一个整数n，求从1到n这n个整数的十进制表示中1出现的次数。
    例如输入12，从1到12这些整数中包含1 的数字有1，10，11和12，1一共出现了5次。
##### 思路
##### Code
```
// 1~n 整数中 1 出现的次数 http://blog.csdn.net/yi_afly/article/details/52012593
func numberOf1Between1AndN(n int) int {
	ans := 0
	for i := 1; i <= n; i *= 10 {
		a := n / i
		b := n % i
		if 0 == a%10 {
			ans += i * a / 10
		} else if 1 == a%10 {
			ans += i*a/10 + b + 1
		} else {
			ans += (a/10 + 1) * i
		}
	}
	return ans
}
```
#### 44.数字序列中某一位的数字
    题目：数字以0123456789101112131415…的格式序列化到一个字符序列中。
    在这个序列中，第5位（从0开始计数）是5，第13位是1，第19位是4，等等。
    请写一个函数求任意位对应的数字。
##### 思路
##### Code
```
func countOfIntegers(digits int) int {
	if 1 == digits {
		return 10
	}
	return int(9 * math.Pow10(digits-1))
}

func beginNumber(digits int) int {
	if 1 == digits {
		return 0
	}
	return int(math.Pow10(digits - 1))
}

func digitAtIndexCore(index, digits int) int {
	number := beginNumber(digits) + index/digits
	indexFromRight := digits - index%digits
	for i := 1; i < indexFromRight; i++ {
		number /= 10
	}
	return number % 10
}

func digitAtIndex(index int) int {
	if 0 > index {
		return -1
	}
	digit := 1
	for {
		numbers := countOfIntegers(digit)
		if index < numbers*digit {
			return digitAtIndexCore(index, digit)
		}
		index -= digit * numbers
		digit++
	}
	return -1
}
```
#### 45.把数组排列成最小的数
    题目：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，
    打印能拼接出的所有数字中最小的一个。例如输入数组{3, 32, 321}，则打印出这3个数字能排成的最小数字321323。
##### 思路
##### Code
```

type intSlice []int

func (p intSlice) Len() int {
	return len(p)
}

func (p intSlice) Less(i, j int) bool {
	return strconv.Itoa(p[i])+strconv.Itoa(p[j]) < strconv.Itoa(p[j])+strconv.Itoa(p[i])
}

func (p intSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func printMinNumber(numbers []int) string {
	if nil == numbers {
		return ""
	}

	sort.Sort(intSlice(numbers))
	var ans string
	for _, value := range numbers {
		ans += strconv.Itoa(value)
	}
	return ans
}
```
#### 46.把数字翻译成字符串
    题目：给定一个数字，我们按照如下规则把它翻译为字符串：0翻译成"a"，
    1翻译成"b"，……，11翻译成"l"，……，25翻译成"z"。一个数字可能有多个翻译。
    例如12258有5种不同的翻译，它们分别是"bccfi"、"bwfi"、"bczi"、"mcfi"和"mzi"。
    请编程实现一个函数用来计算一个数字有多少种不同的翻译方法。
##### 思路
##### Code
```
func getTranslation(number int) int {
	if 0 > number {
		return 0
	}
	return getTranslationCore(strconv.Itoa(number))
}

func getTranslationCore(str string) int {
	length := len(str)
	counts := make([]int, length)
	num := 0

	for i := length - 1; i >= 0; i-- {
		num = 0
		if i < length-1 {
			num = counts[i+1]
		} else {
			num = 1
		}

		if i < length-1 {
			digit1 := str[i] - '0'
			digit2 := str[i+1] - '0'
			number := digit1*10 + digit2
			if 10 <= number && 25 >= number {
				if i < length-2 {
					num += counts[i+2]
				} else {
					num += 1
				}
			}
		}
		counts[i] = num
	}
	return counts[0]
}
```
#### 47.礼物的最大价值
    题目：在一个m×n的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。
    你可以从棋盘的左上角开始拿格子里的礼物，并每次向左或者向下移动一格直到
    到达棋盘的右下角。给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
##### 思路
##### Code
```
func getMaxValue(matrix [][]int) int {
	if nil == matrix {
		return 0
	}

	rows := len(matrix)
	columns := len(matrix[0])
	dp := make([][]int, rows)

	for i := 0; i < rows; i++ {
		dp[i] = make([]int, columns)
		for j := 0; j < columns; j++ {
			if 0 == i && 0 == j {
				dp[i][j] = matrix[i][j]
			}
			if 0 == i && 0 != j {
				dp[i][j] = dp[i][j-1] + matrix[i][j]
			}
			if 0 != i && 0 == j {
				dp[i][j] = dp[i-1][j] + matrix[i][j]
			}
			if 0 != i && 0 != j {
				if dp[i-1][j] >= dp[i][j-1] {
					dp[i][j] = dp[i-1][j] + matrix[i][j]
				} else {
					dp[i][j] = dp[i][j-1] + matrix[i][j]
				}
			}
		}
	}
	return dp[rows-1][columns-1]
}

func getMaxValue2(matrix [][]int) int {
	if nil == matrix {
		return 0
	}

	rows := len(matrix)
	columns := len(matrix[0])
	dp := make([]int, columns)
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			left, up := 0, 0
			if 0 < i {
				up = dp[j]
			}
			if 0 < j {
				left = dp[j-1]
			}

			if left >= up {
				dp[j] = left + matrix[i][j]
			} else {
				dp[j] = up + matrix[i][j]
			}
		}
	}
	return dp[columns-1]
}
```
#### 48.最长不含重复字符的子字符串
    题目：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
    假设字符串中只包含从'a'到'z'的字符。
##### 思路
##### Code
```
func longestSubstring(str string) int {
	curLength, maxLength := 0, 0
	position := make([]int, 26)
	for i := 0; i < 26; i++ {
		position[i] = -1
	}

	for i := 0; i < len(str); i++ {
		preIndex := position[str[i]-'a']
		if 0 > preIndex || i-preIndex > curLength {
			curLength++
		} else {
			if curLength > maxLength {
				maxLength = curLength
			}
			curLength = i - preIndex
		}
		position[str[i]-'a'] = i
		if curLength > maxLength {
			maxLength = curLength
		}
	}
	return maxLength
}
```
#### 49.丑数
    题目：我们把只包含因子2、3和5的数称作丑数（Ugly Number）。求按从小到大的顺序的第1500个丑数。例如6、8都是丑数，但14不是，
    因为它包含因子7。习惯上我们把1当做第一个丑数。
##### 思路
##### Code
```
func getUglyNumber(index int) int {
	if 0 >= index {
		return 0
	}

	uglyNumbers := make([]int, index)
	uglyNumbers[0] = 1
	nextUglyIndex := 1

	var pMultiply2, pMultiply3, pMultiply5 int
	for nextUglyIndex < index {
		uglyNumbers[nextUglyIndex] = min3(uglyNumbers[pMultiply2]*2, uglyNumbers[pMultiply3]*3, uglyNumbers[pMultiply5]*5)

		for uglyNumbers[pMultiply2]*2 <= uglyNumbers[nextUglyIndex] {
			pMultiply2++
		}
		for uglyNumbers[pMultiply3]*3 <= uglyNumbers[nextUglyIndex] {
			pMultiply3++
		}
		for uglyNumbers[pMultiply5]*5 <= uglyNumbers[nextUglyIndex] {
			pMultiply5++
		}
		nextUglyIndex++
	}
	return uglyNumbers[index-1]
}

func min3(num1, num2, num3 int) int {
	number := min(num1, num2)
	return min(num3, number)
}

func min(num1, num2 int) int {
	if num1 < num2 {
		return num1
	}
	return num2
}
```
#### 50.1字符串中第一个只出现一次的字符
    题目：在字符串中找出第一个只出现一次的字符。如输入"abaccdeff"，则输出'b'。
##### 思路
##### Code
```
func firstNoRepeatingChar(str string) byte {
	if 0 >= len(str) {
		return 0
	}

	table := make([]int, 256)
	for i := 0; i < len(str); i++ {
		table[str[i]]++
	}
	for id, value := range table {
		if 1 == value {
			return byte(id)
		}
	}
	return 0
}

```
#### 50.2.字符流中第一个只出现一次的字符
    题目：请实现一个函数用来找出字符流中第一个只出现一次的字符。
    例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是'g'。
    当从该字符流中读出前六个字符"google"时，第一个只出现一次的字符是'l'。
##### 思路
##### Code
```
```
#### 51.数组中的逆序对
    题目：在数组中的两个数字如果前面一个数字大于后面的数字，则这两个数字组成
    一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
##### 思路
##### Code
```
func inversePairs(data []int) int {
	if nil == data || 0 > len(data) {
		return 0
	}

	copy := make([]int, len(data))
	for id, value := range data {
		copy[id] = value
	}

	count := inversePairsCore(data, copy, 0, len(data)-1)
	return count
}

func inversePairsCore(data, copy []int, start, end int) int {
	if start == end {
		copy[start] = data[start]
		return 0
	}
	length := (end - start) / 2
	left := inversePairsCore(copy, data, start, start+length)
	right := inversePairsCore(copy, data, start+length+1, end)

	i := start + length
	j := end
	indexCopy := end
	count := 0

	for i >= start && j >= start+length+1 {
		if data[i] > data[j] {
			copy[indexCopy] = data[i]
			indexCopy--
			i--
			count += j - start - length
		} else {
			copy[indexCopy] = data[j]
			indexCopy--
			j--
		}
	}
	for ; i >= start; i-- {
		copy[indexCopy] = data[i]
		indexCopy--
	}
	for ; j >= start+length+1; j-- {
		copy[indexCopy] = data[j]
		indexCopy--
	}
	return left + right + count
}
```
#### 52.两个链表的第一个公共节点
    题目：输入两个链表，找出它们的第一个公共结点。
##### 思路
##### Code
```
type ListNode struct {
	value int
	next  *ListNode
}

func findFirstCommonNode(head1, head2 *ListNode) *ListNode {
	length1 := getListLength(head1)
	length2 := getListLength(head2)
	var lengthDif int
	var listHeadLong, listHeadShort, firstCommonNode *ListNode
	if length1 >= length2 {
		lengthDif = length1 - length2
		listHeadLong = head1
		listHeadShort = head2
	} else {
		lengthDif = length2 - length1
		listHeadLong = head2
		listHeadShort = head1
	}

	for i := 0; i < lengthDif; i++ {
		listHeadLong = listHeadLong.next
	}
	for listHeadLong != listHeadShort && nil != listHeadLong && nil != listHeadShort {
		listHeadLong = listHeadLong.next
		listHeadShort = listHeadShort.next
	}
	firstCommonNode = listHeadLong
	return firstCommonNode
}

func getListLength(head *ListNode) int {
	var length int
	node := head
	for nil != node {
		length++
		node = node.next
	}
	return length
}
```
#### 53.1.数字在排序数组中出现的次数
    题目：统计一个数字在排序数组中出现的次数。例如输入排序数组{1, 2, 3, 3, 3, 3, 4, 5}和数字3，由于3在这个数组中出现了4次，因此输出4。
##### 思路
##### Code
```
func getFirstK(numbers []int, k, start, end int) int {
	if start > end {
		return -1
	}

	middleIndex := (start + end) / 2
	middleData := numbers[middleIndex]
	if middleData == k {
		if (0 < middleIndex && numbers[middleIndex-1] != k) || 0 == middleIndex {
			return middleIndex
		} else {
			end = middleIndex - 1
		}
	} else if middleData > k {
		end = middleIndex - 1
	} else {
		start = middleIndex + 1
	}
	return getFirstK(numbers, k, start, end)
}

func getLastK(numbers []int, k, start, end int) int {
	if start > end {
		return -1
	}

	middleIndex := (start + end) / 2
	middleData := numbers[middleIndex]
	if middleData == k {
		if (len(numbers)-1 > middleIndex && numbers[middleIndex+1] != k) || len(numbers)-1 == middleIndex {
			return middleIndex
		} else {
			start = middleIndex + 1
		}
	} else if middleData < k {
		start = middleIndex + 1
	} else {
		end = middleIndex - 1
	}
	return getLastK(numbers, k, start, end)
}

func getNumberOfK(numbers []int, k int) int {
	if nil == numbers {
		return 0
	}
	first := getFirstK(numbers, k, 0, len(numbers)-1)
	last := getLastK(numbers, k, 0, len(numbers)-1)
	if first > -1 && last > -1 {
		return last - first + 1
	}
	return 0
}

func getMissingNumber(numbers []int) int {
	if nil == numbers || len(numbers) <= 0 {
		return -1
	}
	left, right := 0, len(numbers)-1
	for left <= right {
		middle := (left + right) >> 1
		if numbers[middle] != middle {
			if 0 == middle || numbers[middle-1] == middle-1 {
				return middle
			}
			right = middle - 1
		} else {
			left = middle + 1
		}
	}
	if left == len(numbers) {
		return len(numbers)
	}
	return -1
}

func getNumberSameAsIndex(numbers []int) int {
	if nil == numbers {
		return -1
	}

	left, right := 0, len(numbers)-1
	for left <= right {
		middle := (left + right) / 2
		if numbers[middle] == middle {
			return middle
		} else if numbers[middle] > middle {
			right = middle - 1
		} else {
			left = middle + 1
		}
	}
	return -1
}
```
#### 53.2.0到n-1中缺失的数字
    题目：一个长度为n-1的递增排序数组中的所有数字都是唯一的，
    并且每个数字都在范围0到n-1之内。在范围0到n-1的n个数字中有且只有一个数字
    不在该数组中，请找出这个数字。
##### 思路
##### Code
```
```
#### 53.3.数组中数值和下标相等的元素
    题目：假设一个单调递增的数组里的每个元素都是整数并且是唯一的。
    请编程实现一个函数找出数组中任意一个数值等于其下标的元素。例如，在数组{-3, -1, 1, 3, 5}中，数字3和它的下标相等。
##### 思路
##### Code
```
```
#### 54.二叉搜索树的第k个节点
    题目：给定一棵二叉搜索树，请找出其中的第k大的结点。
##### 思路
##### Code
```
type BinaryTreeNode struct {
	value       int
	left, right *BinaryTreeNode
}

func kthNode(root *BinaryTreeNode, k int) *BinaryTreeNode {
	if nil == root || k <= 0 {
		return nil
	}

	var target *BinaryTreeNode
	if root.left != nil {
		target = kthNode(root.left, k)
	}

	if target == nil {
		if 1 == k {
			target = root
		} else {
			k--
		}
	}

	if target == nil && root.right != nil {
		target = kthNode(root.right, k)
	}
	return target
}
```
#### 55.1.二叉树的深度
    题目：输入一棵二叉树的根结点，求该树的深度。从根结点到叶结点依次经过的结点
    （含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
##### 思路
##### Code
```
type BinaryTreeNode struct {
	value       int
	left, right *BinaryTreeNode
}

func treeDepth(root *BinaryTreeNode) int {
	if nil == root {
		return 0
	}
	left := treeDepth(root.left)
	right := treeDepth(root.right)
	if left >= right {
		return left + 1
	} else {
		return right + 1
	}
}

func isBalanced(root *BinaryTreeNode, depth *int) bool {
	if nil == root {
		*depth = 0
		return true
	}
	var left, right int
	if isBalanced(root.left, &left) && isBalanced(root.right, &right) {
		diff := left - right
		if -1 <= diff && diff <= 1 {
			if left > right {
				*depth = 1 + left
			} else {
				*depth = 1 + right
			}
			return true
		}
	}
	return false
}

func isBalance(root *BinaryTreeNode) bool {
	var depth int
	return isBalanced(root, &depth)
}
```
#### 55.2平衡二叉树
    题目：输入一棵二叉树的根结点，判断该树是不是平衡二叉树。
    如果某二叉树中任意结点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
##### 思路
##### Code
```
```
#### 56.1.数组中只出现一次的两个数字
    题目：一个整型数组里除了两个数字之外，其他的数字都出现了两次。
    请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
##### 思路
##### Code
```

func findNumbersAppearOnce(data []int) (int, int) {
	if nil == data || len(data) < 2 {
		return 0, 0
	}

	var result int
	for i := 0; i < len(data); i++ {
		result ^= data[i]
	}
	indexOf1 := findFirstBitIs1(result)
	var num1, num2 int
	for j := 0; j < len(data); j++ {
		if isBit1(data[j], indexOf1) {
			num1 ^= data[j]
		} else {
			num2 ^= data[j]
		}
	}
	return num1, num2
}

func findFirstBitIs1(num int) uint {
	var indexBit uint
	for (0 == num&1) && (indexBit < uint(unsafe.Sizeof(int(0))*8)) {
		num = num >> 1
		indexBit++
	}
	return indexBit
}

func isBit1(num int, indexBit uint) bool {
	num = num >> indexBit
	if 1 == num&1 {
		return true
	}
	return false
}

func findNumberAppearOnce2(numbers []int) int {
	if nil == numbers || 0 >= len(numbers) {
		panic(errors.New("invalid input"))
	}

	bitSum := make([]int, 32)
	for i := 0; i < len(numbers); i++ {
		bitMask := 1
		for j := 31; j >= 0; j-- {
			bit := numbers[i] & bitMask
			if 0 != bit {
				bitSum[j] += 1
			}
			bitMask = bitMask << 1
		}
	}

	var result int
	for i := 0; i < 32; i++ {
		result = result << 1
		result += bitSum[i] % 3
	}
	return result
}
```
#### 56.2.数组中唯一只出现一次的数字
    题目：在一个数组中除了一个数字只出现一次之外，其他数字都出现了三次。
    请找出那个吃出现一次的数字。
##### 思路
##### Code
```
```
#### 57.1.和为s的两个数字
    题目：输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。
    如果有多对数字的和等于s，输出任意一对即可。
##### 思路
##### Code
```
func findNumbers(data []int, sum int) (int, int, bool) {
	var found bool
	if nil == data || 1 > len(data) {
		return 0, 0, found
	}

	ahead, behind := 0, len(data)-1
	for ahead < behind {
		currentSum := data[ahead] + data[behind]
		if currentSum == sum {
			found = true
			return data[ahead], data[behind], found
		} else if currentSum < sum {
			ahead++
		} else {
			behind--
		}
	}
	return 0, 0, found
}

func findContinuousSequence(sum int) {
	if 3 > sum {
		return
	}

	middle := (sum + 1) / 2
	small, big := 1, 2
	currentSum := small + big
	for small < middle {
		if currentSum == sum {
			printContinuousSequence(small, big)
		}

		for currentSum > sum && small < middle {
			currentSum -= small
			small++

			if currentSum == sum {
				printContinuousSequence(small, big)
			}
		}
		big++
		currentSum += big
	}
}

func printContinuousSequence(small, big int) {
	for i := small; i <= big; i++ {
		fmt.Printf("%v ", i)
	}
	fmt.Println()
}
```
#### 57.2.为s的连续正数序列
    题目：输入一个正数s，打印出所有和为s的连续正数序列（至少含有两个数）。
    例如输入15，由于1+2+3+4+5=4+5+6=7+8=15，所以结果打印出3个连续序列1～5、4～6和7～8。
##### 思路
##### Code
```
```
#### 58.1.翻转单词顺序
    目：输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。 为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，
    则输出"student. a am I"。
##### 思路
##### Code
```
func reverse(str []rune) {
	copyStr := str

	begin, end := 0, len(copyStr)-1
	for begin < end {
		copyStr[begin], copyStr[end] = copyStr[end], copyStr[begin]
		begin++
		end--
	}
}

func reverseSentence(str string) string {
	if len(str) < 1 {
		return ""
	}

	copyCharArray := []rune(str)
	reverse(copyCharArray)

	var begin, end int
	for end <= len(copyCharArray) {
		if end == len(copyCharArray) {
			reverse(copyCharArray[begin:end])
			break
		}

		if copyCharArray[end] != ' ' {
			end++
		} else {
			reverse(copyCharArray[begin:end])
			end++
			begin = end
		}
	}
	return string(copyCharArray)
}

func leftRotateString(str string, n int) string {
	if len(str) < n {
		return ""
	}

	copyCharArray := []rune(str)
	reverse(copyCharArray[0:n])
	reverse(copyCharArray[n:])
	reverse(copyCharArray[:])

	return string(copyCharArray)
}
```
#### 58.2.左旋转字符串
    题目：字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
    请定义一个函数实现字符串左旋转操作的功能。比如输入字符串"abcdefg"和数字2，
    该函数将返回左旋转2位得到的结果"cdefgab"。
##### 思路
##### Code
```
```
#### 59.1.滑动窗口的最大值
    题目：给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。
    例如，如果输入数组{2, 3, 4, 2, 6, 2, 5, 1}及滑动窗口的大小3，那么一共存在6个滑动窗口，它们的最大值分别为{4, 4, 6, 6, 6, 5}，
##### 思路
##### Code
```
import (
	"gopkg.in/oleiade/lane.v1"
)

func maxInWindows(num []int, size int) []int {
	maxInWindow := lane.NewDeque()

	if len(num) >= size && size >= 1 {
		index := lane.NewDeque()
		for i := 0; i < size; i++ {
			for !index.Empty() && num[i] >= num[index.Last().(int)] {
				index.Pop()
			}
			index.Append(i)
		}
		for i := size; i < len(num); i++ {
			maxInWindow.Append(num[index.First().(int)])
			for !index.Empty() && num[i] >= num[index.Last().(int)] {
				index.Pop()
			}
			if !index.Empty() && index.First().(int) <= i-size {
				index.Shift()
			}
			index.Append(i)
		}
		maxInWindow.Append(num[index.First().(int)])
	}
	var ans []int
	for !maxInWindow.Empty() {
		ans = append(ans, maxInWindow.Shift().(int))
	}
	return ans
}
```
#### 59.2.队列的最大值
    题目：给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。
    例如，如果输入数组{2, 3, 4, 2, 6, 2, 5, 1}及滑动窗口的大小3，那么一共存在6个滑动窗口，它们的最大值分别为{4, 4, 6, 6, 6, 5}，
##### 思路
##### Code
```
```
#### 60.n个骰子的点数
    题目：把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
##### 思路
##### Code
```
```
#### 61.扑克牌的顺子
    题目：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。
    2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王可以看成任意数字。
##### 思路
##### Code
```
import "sort"

func isContinuous(numbers []int) bool {
	if nil == numbers || len(numbers) < 5 {
		return false
	}

	sort.Ints(numbers)
	numberOfZero, numberOfGap := 0, 0
	for _, value := range numbers {
		if value == 0 {
			numberOfZero++
		}
	}

	small, big := 0, 1
	for big < len(numbers) {
		if numbers[small] == numbers[big] {
			return false
		}
		numberOfGap += numbers[big] - numbers[small] - 1
		small = big
		big++
	}
	if numberOfGap > numberOfZero {
		return false
	}
	return true
}
```
#### 62.圆圈中最后剩下的数字
    题目：0, 1, …, n-1这n个数字排成一个圆圈，从数字0开始每次从这个圆圈里删除第m个数字。
    求出这个圆圈里剩下的最后一个数字。
##### 思路
##### Code
```
func lastRemaining(n, m int) int {
	if n < 1 || m < 1 {
		return -1
	}

	last := 0
	for i := 2; i <= n; i++ {
		last = (last + m) % i
	}
	return last
}
```
#### 63.股票的最大利润
    题目：假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖交易该股票可能获得的利润是多少？
    例如一只股票在某些时间节点的价格为{9, 11, 8, 5, 7, 12, 16, 14}。如果我们能在价格为5的时候买入并在价格为16时卖出，则能收获最大的利润11。
##### 思路
##### Code
```
func maxDiff(numbers []int) int {
	if nil == numbers || len(numbers) < 2 {
		return -1
	}

	min := numbers[0]
	max := numbers[1] - min
	for i := 2; i < len(numbers); i++ {
		if numbers[i-1] < min {
			min = numbers[i-1]
		}

		currentDiff := numbers[i] - min
		if currentDiff > max {
			max = currentDiff
		}
	}
	return max
}
```
#### 64.求1+2+...+n
    题目：求1+2+…+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
##### 思路
##### Code
```
```
#### 65.不用加减乘除做加法
    题目：写一个函数，求两个整数之和，要求在函数体内不得使用＋、－、×、÷四则运算符号。
##### 思路
##### Code
```
func Add(num1, num2 int) int {
	var sum, carry int
	for {
		sum = num1 ^ num2
		carry = (num1 & num2) << 1
		num1 = sum
		num2 = carry

		if 0 == num2 {
			break
		}
	}
	return num1
}
```
#### 66.构建乘积数组
    题目：给定一个数组A[0, 1, …, n-1]，请构建一个数组B[0, 1, …, n-1]，
    其中B中的元素B[i] =A[0]×A[1]×… ×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
##### 思路
    n = 10
##### Code
```

```
#### 67.把字符串转换成整数
    题目：请你写一个函数StrToInt，实现把字符串转换成整数这个功能。当然，不能使用atoi或者其他类似的库函数。
##### 思路
##### Code
```
```
#### 68.树中两个节点的最低公共祖先
    题目：输入两个树结点，求它们的最低公共祖先。
##### 思路
##### Code
```
```

